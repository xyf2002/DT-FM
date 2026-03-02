"""GCMA (Genetic Cyclic-Move Algorithm) topology scheduler.

Ported from DeviceEmulator/baselines/gcma.py.  Given a set of devices
with pairwise latency/bandwidth matrices, finds a device-to-pipeline-
stage assignment that minimises a weighted sum of data-parallel
communication cost and pipeline-parallel transfer cost.

Key components:
  * Bipartite matching via ``scipy.optimize.linear_sum_assignment``
  * DP-based open-loop TSP for inter-stage ordering
  * Genetic algorithm with five-point crossover
  * Cyclic-move local-search partitioning refinement

Usage::

    from asteroid.planner.gcma import GCMAScheduler

    sched = GCMAScheduler(
        num_devices=6,
        way=3,               # pipeline stages
        peer_delay=delay_mat,
        peer_bandwidth=bw_mat,
        send_gradient_size=64.0,
        send_activation_size=32.0,
    )
    partitions, scores, history = sched.solve(population_size=50, trails=500)
"""
from __future__ import annotations

import itertools
import logging
import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover – scipy optional at import time
    linear_sum_assignment = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class GCMAScheduler:
    """Genetic Cyclic-Move Algorithm for device-stage assignment.

    Parameters
    ----------
    num_devices:
        Total number of devices/GPUs.
    way:
        Number of pipeline-parallel stages (``pp_size``).
    peer_delay:
        ``(num_devices, num_devices)`` latency matrix (ms).
    peer_bandwidth:
        ``(num_devices, num_devices)`` bandwidth matrix (Gbps).
    send_gradient_size:
        Gradient payload size in MB for DP all-reduce cost.
    send_activation_size:
        Activation payload size in MB for PP transfer cost.
    """

    def __init__(
        self,
        num_devices: int,
        way: int,
        peer_delay: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
        peer_bandwidth: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
        send_gradient_size: float = 64.0,
        send_activation_size: float = 32.0,
    ) -> None:
        self.num_devices = int(max(1, num_devices))
        self.way = int(max(1, way))
        self.partition_size = max(1, self.num_devices // self.way)
        self.dp_size = self.partition_size

        nd = self.num_devices
        if peer_delay is not None:
            self.peer_delay = np.asarray(peer_delay, dtype=np.float64)
        else:
            self.peer_delay = np.ones((nd, nd), dtype=np.float64)
            np.fill_diagonal(self.peer_delay, 0.0)

        if peer_bandwidth is not None:
            self.peer_bandwidth = np.asarray(peer_bandwidth, dtype=np.float64)
        else:
            self.peer_bandwidth = np.full((nd, nd), 10.0, dtype=np.float64)

        self.send_gradient_size = float(max(0.0, send_gradient_size))
        self.send_activation_size = float(max(0.0, send_activation_size))

    # ── Link-time helpers ───────────────────────────────────────────────────

    def _safe_link_time(self, src: int, dst: int, payload_mb: float) -> float:
        """Transfer time in seconds: delay + payload / bandwidth."""
        if src == dst:
            return 0.0
        delay_s = float(self.peer_delay[src, dst]) / 1000.0
        bw_gbps = float(max(self.peer_bandwidth[src, dst], 1e-9))
        return delay_s + (payload_mb * 8.0) / (bw_gbps * 1000.0)

    # ── Cost functions ──────────────────────────────────────────────────────

    def compute_data_parallel_cost(
        self,
        partitions: list[tuple[int, ...]],
    ) -> float:
        """Sum of link times within each partition (DP gradient exchange)."""
        total = 0.0
        for part in partitions:
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    total += self._safe_link_time(
                        part[i], part[j], self.send_gradient_size,
                    )
        return total

    def compute_pipeline_parallel_cost(
        self,
        partitions: list[tuple[int, ...]],
    ) -> tuple[float, list[int], list[list[float]]]:
        """Pipeline cost via bipartite matching + DP-TSP ordering.

        Returns
        -------
        cost:
            Total pipeline-parallel transfer cost.
        best_path:
            Best stage ordering found by TSP.
        match_matrix:
            Cost matrix used for bipartite matching.
        """
        if linear_sum_assignment is None:
            raise ImportError(
                "scipy is required for GCMA pipeline-parallel cost "
                "computation. Install it: pip install scipy"
            )

        way = len(partitions)
        if way <= 1:
            return 0.0, list(range(way)), [[0.0]]

        psz = self.partition_size

        # Build cost matrix between every pair of stages
        match_matrix: list[list[float]] = []
        for i in range(way):
            row: list[float] = []
            for j in range(way):
                if i == j:
                    row.append(0.0)
                else:
                    cost_mat = np.zeros((psz, psz), dtype=np.float64)
                    for si in range(min(psz, len(partitions[i]))):
                        for sj in range(min(psz, len(partitions[j]))):
                            cost_mat[si, sj] = self._safe_link_time(
                                partitions[i][si],
                                partitions[j][sj],
                                self.send_activation_size,
                            )
                    ri, ci = linear_sum_assignment(cost_mat)
                    row.append(float(cost_mat[ri, ci].sum()))
                match_matrix.append(row)

        # DP-based open-loop TSP with bitmask memo
        dist = np.array(match_matrix, dtype=np.float64)
        full_mask = (1 << way) - 1
        dp_tsp: dict[tuple[int, int], float] = {}
        parent: dict[tuple[int, int], int] = {}

        for start in range(way):
            dp_tsp[(1 << start, start)] = 0.0

        for mask in range(1, full_mask + 1):
            for last in range(way):
                if not (mask & (1 << last)):
                    continue
                prev_mask = mask ^ (1 << last)
                if prev_mask == 0:
                    continue
                for prev in range(way):
                    if not (prev_mask & (1 << prev)):
                        continue
                    key_prev = (prev_mask, prev)
                    if key_prev not in dp_tsp:
                        continue
                    new_cost = dp_tsp[key_prev] + dist[prev, last]
                    key_cur = (mask, last)
                    if key_cur not in dp_tsp or new_cost < dp_tsp[key_cur]:
                        dp_tsp[key_cur] = new_cost
                        parent[key_cur] = prev

        # Recover best path
        best_cost = float("inf")
        best_path: list[int] = list(range(way))
        for last in range(way):
            key = (full_mask, last)
            if key in dp_tsp and dp_tsp[key] < best_cost:
                best_cost = dp_tsp[key]
                # Traceback
                path = [last]
                cur_mask = full_mask
                cur_node = last
                while (cur_mask, cur_node) in parent:
                    prev_node = parent[(cur_mask, cur_node)]
                    cur_mask ^= (1 << cur_node)
                    cur_node = prev_node
                    path.append(cur_node)
                path.reverse()
                best_path = path

        return float(best_cost), best_path, match_matrix

    # ── Main solve (GCMA evolutionary search) ───────────────────────────────

    def solve(
        self,
        population_size: int = 100,
        trails: int = 4900,
        mode: str = "default",
    ) -> tuple[list[list[int]], list[float], list[float]]:
        """Run GCMA evolutionary search.

        Parameters
        ----------
        population_size:
            Initial population count.
        trails:
            Number of evolutionary iterations.
        mode:
            ``"default"`` or ``"baseline"`` gain calculation mode.

        Returns
        -------
        partitions:
            All partition orderings explored.
        scores:
            Score per partition.
        min_scores:
            Running minimum score history.
        """
        nd = self.num_devices
        way = self.way
        psz = self.partition_size

        def _to_partition_list(flat: list[int]) -> list[tuple[int, ...]]:
            return [
                tuple(flat[idx: idx + psz])
                for idx in range(0, nd, psz)
            ]

        def _five_point_crossover(
            parent1: list[int],
            parent2: list[int],
        ) -> list[int]:
            p1_str = [0] * nd
            p2_str = [0] * nd
            for i in range(nd):
                p1_str[parent1[i]] = i // psz
                p2_str[parent2[i]] = i // psz

            points = list(range(nd))
            random.shuffle(points)
            points = points[:5]
            for pt in points:
                p2_str[pt] = p1_str[pt]

            sizes = [0] * way
            for pidx in p2_str:
                sizes[pidx] += 1

            for i in range(nd):
                cur_p = p2_str[i]
                if sizes[cur_p] <= psz:
                    continue
                target = None
                for j in range(way):
                    if sizes[j] < psz:
                        target = j
                        break
                if target is None:
                    continue
                sizes[cur_p] -= 1
                sizes[target] += 1
                p2_str[i] = target

            return p2_str

        def _cyclic_partitioning(offspring_str: list[int]) -> list[int]:
            """Cyclic-move local search refinement."""

            def _calculate_gain_default(
                cur_off: list[int],
                locked_v: list[int],
            ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[list[int | None]]]:
                sizes = [0] * way
                for pidx in cur_off:
                    sizes[pidx] += 1

                gain = np.zeros((nd, way), dtype=float)
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] != 0:
                        continue
                    gain[v, pidx] = np.inf
                    for t, tpidx in enumerate(cur_off):
                        pp_cost = self._safe_link_time(v, t, self.send_activation_size)
                        if pidx != tpidx:
                            den = max(1, sizes[tpidx])
                            gain[v, tpidx] += pp_cost / den
                        elif v != t and gain[v, tpidx] > pp_cost:
                            gain[v, tpidx] = pp_cost

                g_i = np.full(way, np.inf, dtype=float)
                g_i_trace: list[list[int | None]] = [[None, None] for _ in range(way)]
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] != 0:
                        continue
                    if gain[v, pidx] < g_i[pidx]:
                        g_i[pidx] = gain[v, pidx]
                        g_i_trace[pidx][0] = v

                g_i = np.full(way, -np.inf, dtype=float)
                g_ij = np.full((way, way), -np.inf, dtype=float)
                for pidx, trace in enumerate(g_i_trace):
                    v = trace[0]
                    if v is None:
                        continue
                    for tpidx, tgain in enumerate(gain[v]):
                        if tpidx == pidx:
                            continue
                        tgain_net = tgain - gain[v, pidx]
                        if tgain_net > g_ij[pidx, tpidx]:
                            g_ij[pidx, tpidx] = tgain_net
                        if tgain_net > g_i[pidx]:
                            g_i[pidx] = tgain_net
                            g_i_trace[pidx] = [v, tpidx]
                return g_ij, g_i, g_i_trace

            def _calculate_gain_baseline(
                cur_off: list[int],
                locked_v: list[int],
            ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[list[int | None]]]:
                gain = np.zeros((nd, way), dtype=float)
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] != 0:
                        continue
                    for t, tpidx in enumerate(cur_off):
                        pp_cost = self._safe_link_time(v, t, self.send_activation_size)
                        dp_cost = self._safe_link_time(v, t, self.send_gradient_size)
                        if v != t:
                            gain[v, tpidx] += pp_cost
                            gain[v, tpidx] -= dp_cost

                g_i_trace: list[list[int | None]] = [[None, None] for _ in range(way)]
                g_i = np.full(way, -np.inf, dtype=float)
                g_ij = np.full((way, way), -np.inf, dtype=float)
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] != 0:
                        continue
                    for tpidx, tgain in enumerate(gain[v]):
                        if tpidx == pidx:
                            continue
                        tgain_net = tgain - gain[v, pidx]
                        if tgain_net > g_ij[pidx, tpidx]:
                            g_ij[pidx, tpidx] = tgain_net
                        if tgain_net > g_i[pidx]:
                            g_i[pidx] = tgain_net
                            g_i_trace[pidx] = [v, tpidx]
                return g_ij, g_i, g_i_trace

            def _move_cycles(off_str: list[int]) -> list[int]:
                sums = [0.0]
                locked_part = [0] * way
                locked_v = [0] * nd
                offsprings = [off_str]

                for _ in range(way):
                    cur = offsprings[-1].copy()
                    movements: list[tuple[int, int, int]] = []
                    epsilon: list[float] = []
                    tau: list[float] = []

                    if mode == "default":
                        g_ij, g_i, g_i_trace = _calculate_gain_default(cur, locked_v)
                    else:
                        g_ij, g_i, g_i_trace = _calculate_gain_baseline(cur, locked_v)

                    s0 = int(np.argmax(g_i))
                    si = s0

                    for _ in range(nd):
                        v, pv = g_i_trace[si]
                        if v is None:
                            if not movements:
                                break
                            v = movements[-1][0]
                            pv = s0
                        if pv is None:
                            break

                        cur[v] = pv
                        locked_v[v] = 1
                        locked_part[pv] = 1
                        movements.append((v, si, pv))
                        epsilon.append(float(g_i[si]))
                        tau.append(float(g_ij[si, s0]))
                        si = pv
                        if si == s0:
                            break
                        if mode == "default":
                            g_ij, g_i, g_i_trace = _calculate_gain_default(cur, locked_v)
                        else:
                            g_ij, g_i, g_i_trace = _calculate_gain_baseline(cur, locked_v)

                    if not movements:
                        break

                    max_sum = 0.0
                    best_l = 0
                    for idx in range(1, len(epsilon)):
                        val = float(np.sum(epsilon[:idx])) + tau[idx]
                        if val > max_sum:
                            max_sum = val
                            best_l = idx

                    for idx in range(len(epsilon) - 1, best_l, -1):
                        cur[movements[idx][0]] = movements[idx][1]
                    cur[movements[best_l][0]] = s0
                    offsprings.append(cur)
                    sums.append(max_sum)

                    if sum(locked_part) == len(locked_part):
                        break

                max_sum_total = 0.0
                best_m = 0
                for idx in range(1, len(sums)):
                    val = float(np.sum(sums[:idx]))
                    if val > max_sum_total:
                        max_sum_total = val
                        best_m = idx - 1
                return offsprings[best_m]

            for _ in range(1):
                offspring_str = _move_cycles(offspring_str)
            return offspring_str

        # ── Population initialisation ───────────────────────────────────────
        nodes = list(range(nd))
        partitions: list[list[int]] = []
        scores: list[float] = []
        min_scores: list[float] = []

        for i in range(population_size):
            cur = nodes.copy()
            rng = random.Random(i)
            rng.shuffle(cur)
            partitions.append(cur)

        for part in partitions:
            cp = _to_partition_list(part)
            dp_cost = self.compute_data_parallel_cost(cp)
            pp_cost, _, _ = self.compute_pipeline_parallel_cost(cp)
            scores.append(dp_cost + 2 * pp_cost)
            min_scores.append(float(np.min(scores)))

        # ── Evolutionary loop ───────────────────────────────────────────────
        for i in range(trails):
            np.random.seed(i)
            p1_idx, p2_idx = np.random.randint(population_size, size=2).tolist()

            ga_off = _five_point_crossover(partitions[p1_idx], partitions[p2_idx])
            off_str = _cyclic_partitioning(ga_off)

            off_flat: list[list[int]] = [[] for _ in range(way)]
            for v_idx, pidx in enumerate(off_str):
                off_flat[pidx].append(v_idx)

            off_cp = [tuple(group) for group in off_flat]
            off_dp_cost = self.compute_data_parallel_cost(off_cp)
            off_pp_cost, _, _ = self.compute_pipeline_parallel_cost(off_cp)
            off_score = off_dp_cost + 2 * off_pp_cost
            off_list = list(itertools.chain.from_iterable(off_flat))

            if off_score > max(scores[p1_idx], scores[p2_idx]):
                partitions.append(off_list)
                scores.append(off_score)
            else:
                replaced = p1_idx if scores[p1_idx] > scores[p2_idx] else p2_idx
                old_part = partitions[replaced]
                partitions[replaced] = off_list
                partitions.append(old_part)

                old_score = scores[replaced]
                scores[replaced] = off_score
                scores.append(old_score)

            min_scores.append(float(np.min(scores)))

        return partitions, scores, min_scores


__all__ = ["GCMAScheduler"]
