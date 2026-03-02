from __future__ import annotations

import itertools
import logging
import random

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .dp_partitioner import DPPartitioner

_LOGGER = logging.getLogger(__name__)


class GCMAScheduler:
    """GCMA evolutionary solver for heterogeneous GPU topology.

    Stage 1 — GCMA: Genetic Crossover + Multi-cycle Assignment
      Decides which GPUs form each pipeline group and DP replica group.
      Combines:
        - Bipartite matching (Hungarian algorithm)
        - Open-loop TSP (DP-based)
        - Evolutionary population search

    Stage 2 — DP partition (uses DPPartitioner)
    """

    def __init__(
        self,
        num_devices: int,
        pp_size: int,
        dp_size: int,
        peer_delay: NDArray[np.float64],
        peer_bandwidth: NDArray[np.float64],
        send_gradient_size: float,
        send_activation_size: float,
    ) -> None:
        self.num_devices = int(num_devices)
        self.way = int(pp_size)
        self.partition_size = int(dp_size)
        self.peer_delay = np.asarray(peer_delay, dtype=float)
        self.peer_bandwidth = np.asarray(peer_bandwidth, dtype=float)
        self.send_gradient_size = float(send_gradient_size)
        self.send_activation_size = float(send_activation_size)

        if self.num_devices != self.way * self.partition_size:
            raise ValueError(
                "num_devices must equal pp_size * dp_size: "
                f"{self.num_devices} != {self.way} * "
                f"{self.partition_size}"
            )

    def _safe_link_time(self, src: int, dst: int, payload_gb: float) -> float:
        bw = self.peer_bandwidth[src, dst]
        if bw <= 0:
            return float("inf")
        return self.peer_delay[src, dst] / 1e3 + payload_gb * 8 / bw

    def compute_data_parallel_cost(
        self,
        candidate_partition: list[tuple[int, ...]] | list[list[int]],
    ) -> float:
        data_parallel_cost = float("-inf")
        payload_gb = self.send_gradient_size / max(1, self.partition_size)

        for partition in candidate_partition:
            within_cost = [0.0] * self.partition_size
            for i in range(self.partition_size):
                for j in range(self.partition_size):
                    if i == j:
                        continue
                    src = partition[i]
                    dst = partition[j]
                    within_cost[i] += 2 * self._safe_link_time(
                        src,
                        dst,
                        payload_gb,
                    )
            data_parallel_cost = max(data_parallel_cost, max(within_cost))

        if data_parallel_cost == float("-inf"):
            return float("inf")
        return float(data_parallel_cost)

    def compute_pipeline_parallel_cost(
        self,
        candidate_partition: list[tuple[int, ...]] | list[list[int]],
    ) -> tuple[float, list[int], list[list[list[tuple[int, int]] | None]]]:
        way = self.way
        psz = self.partition_size

        def bipartite_matching(
            part_0: tuple[int, ...] | list[int],
            part_1: tuple[int, ...] | list[int],
        ) -> tuple[float, list[tuple[int, int]]]:
            cost_mat = np.zeros((psz, psz), dtype=float)
            for i in range(psz):
                for j in range(psz):
                    src = part_0[i]
                    dst = part_1[j]
                    cost_mat[i, j] = self._safe_link_time(
                        src,
                        dst,
                        self.send_activation_size,
                    )

            descending = np.argsort(cost_mat.flatten())[::-1]
            inf_weight = 1e6
            for idx in descending:
                row = int(idx // psz)
                col = int(idx % psz)
                cur_max = float(cost_mat[row, col])
                cost_mat[row, col] = inf_weight
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                if float(cost_mat[row_ind, col_ind].sum()) >= inf_weight:
                    return cur_max, list(
                        zip(row_ind.tolist(), col_ind.tolist())
                    )

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            final_cost = float(np.max(cost_mat[row_ind, col_ind]))
            return final_cost, list(zip(row_ind.tolist(), col_ind.tolist()))

        cross_cost = np.zeros((way, way), dtype=float)
        match_matrix: list[list[list[tuple[int, int]] | None]] = [
            [None] * way for _ in range(way)
        ]

        for i in range(way):
            for j in range(i + 1, way):
                cost, match = bipartite_matching(
                    candidate_partition[i],
                    candidate_partition[j],
                )
                cross_cost[i, j] = cost
                cross_cost[j, i] = cost
                match_matrix[i][j] = match
                match_matrix[j][i] = [(c, r) for r, c in match]

        best_cost = float("inf")
        best_path: list[int] = []

        for start in range(way):
            dp_table = np.full((way, 1 << way), np.inf, dtype=float)
            trace = np.zeros((way, 1 << way), dtype=int)

            def _bitmask(nodes: list[int]) -> int:
                mask = 0
                for node in nodes:
                    mask |= 1 << node
                return mask

            def _solve(node: int, future: list[int]) -> float:
                if not future:
                    return 0.0

                bm = _bitmask(future)
                cached = dp_table[node, bm]
                if cached < np.inf:
                    return float(cached)

                best_d = np.inf
                best_next = future[0]
                for nxt in future:
                    nxt_future = [f for f in future if f != nxt]
                    nxt_bm = _bitmask(nxt_future)
                    if dp_table[nxt, nxt_bm] == np.inf:
                        d = cross_cost[node, nxt] + _solve(nxt, nxt_future)
                    else:
                        d = cross_cost[node, nxt] + dp_table[nxt, nxt_bm]
                    if d < best_d:
                        best_d = d
                        best_next = nxt

                dp_table[node, bm] = best_d
                trace[node, bm] = best_next
                return float(best_d)

            future = [node for node in range(way) if node != start]
            cost = _solve(start, future)
            if cost < best_cost:
                best_cost = cost
                path = [start]
                cur = start
                remaining = list(future)
                while remaining:
                    bm = _bitmask(remaining)
                    nxt = int(trace[cur, bm])
                    path.append(nxt)
                    remaining.remove(nxt)
                    cur = nxt
                best_path = path

        return float(best_cost), best_path, match_matrix

    def gcma(
        self,
        population_size: int = 100,
        trails: int = 4900,
        mode: str = "default",
    ) -> tuple[list[list[int]], list[float], list[float]]:
        nd = self.num_devices
        way = self.way
        psz = self.partition_size

        def _to_partition_list(flat: list[int]) -> list[tuple[int, ...]]:
            return [
                tuple(flat[idx : idx + psz])
                for idx in range(0, nd, psz)
            ]

        def five_point_crossover(
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

        def cyclic_partitioning(offspring_str: list[int]) -> list[int]:
            def calculate_gain_default(
                cur_off: list[int],
                locked_v: list[int],
            ) -> tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                list[list[int | None]],
            ]:
                sizes = [0] * way
                for pidx in cur_off:
                    sizes[pidx] += 1

                gain = np.zeros((nd, way), dtype=float)
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] != 0:
                        continue
                    gain[v, pidx] = np.inf
                    for t, tpidx in enumerate(cur_off):
                        pp_cost = self._safe_link_time(
                            v,
                            t,
                            self.send_activation_size,
                        )
                        if pidx != tpidx:
                            den = max(1, sizes[tpidx])
                            gain[v, tpidx] += pp_cost / den
                        elif v != t and gain[v, tpidx] > pp_cost:
                            gain[v, tpidx] = pp_cost

                g_i = np.full(way, np.inf, dtype=float)
                g_i_trace: list[list[int | None]] = [
                    [None, None] for _ in range(way)
                ]
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

            def calculate_gain_baseline(
                cur_off: list[int],
                locked_v: list[int],
            ) -> tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                list[list[int | None]],
            ]:
                gain = np.zeros((nd, way), dtype=float)
                for v, pidx in enumerate(cur_off):
                    if locked_v[v] != 0:
                        continue
                    for t, tpidx in enumerate(cur_off):
                        pp_cost = self._safe_link_time(
                            v,
                            t,
                            self.send_activation_size,
                        )
                        dp_cost = self._safe_link_time(
                            v,
                            t,
                            self.send_gradient_size,
                        )
                        if v != t:
                            gain[v, tpidx] += pp_cost
                            gain[v, tpidx] -= dp_cost

                g_i_trace: list[list[int | None]] = [
                    [None, None] for _ in range(way)
                ]
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

            def move_cycles(off_str: list[int]) -> list[int]:
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
                        g_ij, g_i, g_i_trace = calculate_gain_default(
                            cur,
                            locked_v,
                        )
                    else:
                        g_ij, g_i, g_i_trace = calculate_gain_baseline(
                            cur,
                            locked_v,
                        )

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
                            g_ij, g_i, g_i_trace = calculate_gain_default(
                                cur,
                                locked_v,
                            )
                        else:
                            g_ij, g_i, g_i_trace = calculate_gain_baseline(
                                cur,
                                locked_v,
                            )

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

                max_sum = 0.0
                best_m = 0
                for idx in range(1, len(sums)):
                    val = float(np.sum(sums[:idx]))
                    if val > max_sum:
                        max_sum = val
                        best_m = idx - 1
                return offsprings[best_m]

            for _ in range(1):
                offspring_str = move_cycles(offspring_str)
            return offspring_str

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

        for i in range(trails):
            np.random.seed(i)
            p1_idx, p2_idx = np.random.randint(
                population_size,
                size=2,
            ).tolist()

            ga_off = five_point_crossover(
                partitions[p1_idx],
                partitions[p2_idx],
            )
            off_str = cyclic_partitioning(ga_off)

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

    def get_pipelines(
        self,
        candidate_partition: list[tuple[int, ...]] | list[list[int]],
        path: list[int],
        match_matrix: list[list[list[tuple[int, int]] | None]],
    ) -> NDArray[np.int_]:
        way = self.way
        psz = self.partition_size
        pipeline = np.zeros((way, psz), dtype=int)

        for stage_idx, part_idx in enumerate(path):
            if way == 1:
                pipeline[stage_idx, :] = np.arange(psz)
                continue

            if stage_idx > 0:
                last_part_idx = path[stage_idx - 1]
                bm = match_matrix[last_part_idx][part_idx] or []
                for match in bm:
                    src_local, dst_local = match
                    for i in range(psz):
                        if pipeline[stage_idx - 1, i] == src_local:
                            pipeline[stage_idx, i] = dst_local
            else:
                next_part_idx = path[1]
                bm = match_matrix[part_idx][next_part_idx] or []
                for i, match in enumerate(bm):
                    pipeline[0, i] = match[0]

        for stage_idx, part_idx in enumerate(path):
            for i in range(psz):
                local_idx = int(pipeline[stage_idx, i])
                pipeline[stage_idx, i] = candidate_partition[part_idx][
                    local_idx
                ]

        return pipeline

    def _build_gpu_map(
        self,
        pipeline_matrix: NDArray[np.int_],
    ) -> dict[int, int]:
        gpu_map: dict[int, int] = {}
        for dp_rank in range(self.partition_size):
            for pp_rank in range(self.way):
                global_rank = dp_rank * self.way + pp_rank
                gpu_map[global_rank] = int(pipeline_matrix[pp_rank, dp_rank])
        return gpu_map

    def solve(self) -> dict[str, object]:
        partitions, scores, min_scores = self.gcma()

        best_idx = int(np.argmin(scores))
        best_flat = partitions[best_idx]
        best_cp = [
            tuple(best_flat[idx : idx + self.partition_size])
            for idx in range(0, self.num_devices, self.partition_size)
        ]

        dp_cost = self.compute_data_parallel_cost(best_cp)
        pp_cost, pp_path, pp_match = self.compute_pipeline_parallel_cost(
            best_cp
        )
        total_cost = dp_cost + 2 * pp_cost
        pipeline_matrix = self.get_pipelines(best_cp, pp_path, pp_match)
        gpu_map = self._build_gpu_map(pipeline_matrix)

        result: dict[str, object] = {
            "best_partition": [list(group) for group in best_cp],
            "pipeline_path": pp_path,
            "pipeline_matrix": pipeline_matrix.tolist(),
            "gpu_map": gpu_map,
            "data_parallel_cost": dp_cost,
            "pipeline_parallel_cost": pp_cost,
            "total_cost": total_cost,
            "scores": scores,
            "min_scores": min_scores,
        }

        num_layers = getattr(self, "num_layers", None)
        profiler_data = getattr(self, "profiler_data", None)
        if isinstance(num_layers, int) and num_layers > 0:
            partitioner = DPPartitioner(
                num_layers=num_layers,
                num_devices=self.way,
                profiler_data=profiler_data,
            )
            result["layer_partition_points"] = partitioner.partition(
                is_average=True
            )

        _LOGGER.info(
            "GCMA solved. total=%.6f dp=%.6f pp=%.6f",
            total_cost,
            dp_cost,
            pp_cost,
        )
        return result


__all__ = ["GCMAScheduler"]
