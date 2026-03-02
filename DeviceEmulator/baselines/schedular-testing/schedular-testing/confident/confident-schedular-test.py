import json
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimulatedProfiler:
    """Mock profiler that loads hardware and compute parameters from a dictionary."""
    def __init__(self, config: dict):
        self.num_devices = config["num_devices"]
        self.num_layers = config["num_layers"]
        
        # Original code tracks profiles per device. Here we use the baseline 
        # and rely on computing_capacities to scale them during scheduling.
        self.layer_profiles = {
            d: config["layer_profiles"] for d in range(self.num_devices)
        }
        
        self.output_sizes = config["output_sizes"]
        self.bandwidths = config["bandwidths"]
        self.computing_capacities = config["computing_capacities"]
        self.available_memory = config["available_memory"]
        
        # Precompute time intervals like the original ConfidantProfiler
        self.time_intervals = {d: {} for d in range(self.num_devices)}
        for d in range(self.num_devices):
            self.build_time_intervals(d, self.layer_profiles[d])

    def build_time_intervals(self, device_id: int, profiles: List[dict]):
        n = len(profiles)
        for s in range(n):
            for e in range(s, n):
                fwd = sum(profiles[i]['forward_ms'] for i in range(s, e + 1))
                bwd = sum(profiles[i]['backward_ms'] for i in range(s, e + 1))
                self.time_intervals[device_id][(s, e, 0)] = fwd
                self.time_intervals[device_id][(s, e, 1)] = bwd

    # --- Offline data getters (Used by Scheduler) ---
    def get_time_interval(self, device_id: int, start: int, end: int, phase: int) -> float:
        return self.time_intervals.get(device_id, {}).get((start, end, phase), 0.0)

    def get_output_size(self, layer_idx: int) -> float:
        return self.output_sizes[layer_idx] if layer_idx < len(self.output_sizes) else 0.0

    def get_bandwidth(self, device_id: int) -> float:
        return self.bandwidths[device_id] if device_id < len(self.bandwidths) else 1.0

    def get_computing_capacity(self, device_id: int) -> float:
        return self.computing_capacities[device_id] if device_id < len(self.computing_capacities) else 1.0

    def get_available_memory(self, device_id: int) -> float:
        return self.available_memory[device_id] if device_id < len(self.available_memory) else 0.0


class ConfidantScheduler:
    """The DP-based dynamic scheduler extracted exactly from your source code."""
    def __init__(self, profiler: SimulatedProfiler, num_layers: int, num_devices: int):
        self.profiler = profiler
        self.num_layers = num_layers
        self.num_devices = num_devices

    def _get_time(self, device_id: int, start: int, end: int) -> float:
        fwd = self.profiler.get_time_interval(device_id, start, end, 0)
        bwd = self.profiler.get_time_interval(device_id, start, end, 1)
        return fwd + bwd

    def _get_comm_time(self, layer_idx: int, device_id: int) -> float:
        output_size = self.profiler.get_output_size(layer_idx)
        bandwidth = self.profiler.get_bandwidth(device_id)
        return output_size / bandwidth if bandwidth > 0 else 0.0

    def calculate_partition_point(self, is_average: bool = True) -> List[int]:
        n = self.num_layers
        k = self.num_devices

        logger.info(f"Computing partition: {n} layers, {k} devices")

        if is_average:
            capacities = [self.profiler.get_computing_capacity(d) for d in range(k)]
        else:
            capacities = [1.0] * k

        # DP: dp[i][j] = min bottleneck time to assign layers 0..i across devices 0..j
        INF = float('inf')
        dp = [[INF] * k for _ in range(n)]
        split = [[-1] * k for _ in range(n)]

        # Base case: all layers 0..i on device 0
        for i in range(n):
            dp[i][0] = self._get_time(0, 0, i) / capacities[0]

        # Fill DP table
        for j in range(1, k):
            for i in range(j, n):
                for m in range(j - 1, i):
                    # Layers 0..m on devices 0..j-1, layers m+1..i on device j
                    compute_time = self._get_time(j, m + 1, i) / capacities[j]
                    comm_time = self._get_comm_time(m, j - 1)
                    cost = max(dp[m][j - 1], compute_time + comm_time)

                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        # Backtrack to find partition points
        points = []
        i, j = n - 1, k - 1
        while j > 0:
            m = split[i][j]
            points.append(m)
            i = m
            j -= 1
        points.reverse()

        bottleneck = dp[n - 1][k - 1]
        logger.info(f"Partition: {points}, Bottleneck: {bottleneck:.2f}ms")
        return points

if __name__ == "__main__":
    # Load the simulated configuration
    with open("./confident_config.json", "r") as f:
        config = json.load(f)

    # Initialize Simulator
    profiler = SimulatedProfiler(config)
    scheduler = ConfidantScheduler(profiler, config["num_layers"], config["num_devices"])

    # Run Scheduler
    print("\n--- Running Scheduler Simulation ---")
    partition_points = scheduler.calculate_partition_point(is_average=True)
    
    print("\n--- Results ---")
    print(f"Final Partition Points: {partition_points}")
    print("Interpretation: ")
    
    # Dynamically print the partitions for ANY number of devices
    for i in range(len(partition_points)):
        start_layer = 0 if i == 0 else partition_points[i-1] + 1
        end_layer = partition_points[i]
        device_name = config['simulated_hardware'][i] if 'simulated_hardware' in config else f"GPU {i}"
        print(f"  Device {i} ({device_name}) gets layers: {start_layer} to {end_layer}")
        
    # Print the last device
    last_device_idx = len(partition_points)
    start_layer = partition_points[-1] + 1
    end_layer = config['num_layers'] - 1
    device_name = config['simulated_hardware'][-1] if 'simulated_hardware' in config else f"GPU {last_device_idx}"
    print(f"  Device {last_device_idx} ({device_name}) gets layers: {start_layer} to {end_layer}")