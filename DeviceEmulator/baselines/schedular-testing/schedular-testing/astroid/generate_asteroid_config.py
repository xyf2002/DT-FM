import json
import random
import os
import pickle
import argparse

# A mix of Flagship, Mid-range, and weak IoT devices
HARDWARE_POOL = {
    "Snapdragon 8 Elite Gen 5": 16226.0,
    "Snapdragon 8 Gen 3": 7113.0,
    "Apple A18 Pro": 3790.0,
    "Snapdragon 680 (IoT)": 95.0,
    "Nvidia Tegra X1 (IoT)": 60.0,
    "Rockchip RK3588 (IoT)": 240.0
}

def parse_fedscale_network_trace(filepath="client_device_capacity"):
    """Reads the FedScale client_device_capacity pickle file."""
    bandwidths_mbs = []
    if os.path.exists(filepath):
        print(f"Reading MobiPerf trace from {filepath}...")
        try:
            with open(filepath, 'rb') as f:
                client_data = pickle.load(f)
            if isinstance(client_data, dict):
                for client_id, metrics in client_data.items():
                    comm_kbps = metrics.get('communication', 0.0)
                    if comm_kbps > 0:
                        bandwidths_mbs.append(float(comm_kbps) / 8192.0)
            elif isinstance(client_data, list):
                for item in client_data:
                    if isinstance(item, (int, float)):
                        bandwidths_mbs.append(float(item) / 8192.0)
        except Exception as e:
            print(f"⚠️ Failed to parse the pickle file: {e}")
            
    if not bandwidths_mbs:
        print("⚠️ No valid network data extracted. Using log-normal fallback...")
        bandwidths_mbs = [max(1.0, random.lognormvariate(2.5, 0.8)) for _ in range(100)]
        
    return bandwidths_mbs

def generate_asteroid_config(total_layers, min_devices, max_devices, num_stages, distribution):
    MAX_SCORE = HARDWARE_POOL["Snapdragon 8 Elite Gen 5"]
    
    # Randomly pick a cluster size between the min and max provided
    num_devices = random.randint(min_devices, max_devices)
    
    # --- DISTRIBUTION LOGIC ---
    if distribution == "balanced":
        # Homogeneous cluster (only flagship / high-end devices)
        top_tier = ["Snapdragon 8 Elite Gen 5", "Snapdragon 8 Gen 3", "Apple A18 Pro"]
        base_device_types = random.choices(top_tier, k=num_devices)
    elif distribution == "skewed":
        # Extreme heterogeneity: exactly 1 Flagship, N-1 IoT devices
        base_device_types = ["Snapdragon 8 Elite Gen 5"]
        iot_tier = ["Snapdragon 680 (IoT)", "Nvidia Tegra X1 (IoT)", "Rockchip RK3588 (IoT)"]
        base_device_types += random.choices(iot_tier, k=num_devices - 1)
        random.shuffle(base_device_types)  # Shuffle so the flagship isn't always Node 0
    else:
        # 'random' - A completely uniform mix of the entire pool
        base_device_types = random.choices(list(HARDWARE_POOL.keys()), k=num_devices)
    # --------------------------
    
    # Make names unique for the logger
    selected_names = [f"{name} (Node {i})" for i, name in enumerate(base_device_types)]
    
    network_pool = parse_fedscale_network_trace("client_device_capacity")
    selected_bandwidths = random.choices(network_pool, k=num_devices) 
    
    computing_capacities = []
    available_memory_mb = []
    
    for base_name in base_device_types:
        score = HARDWARE_POOL[base_name]
        computing_capacities.append(score / MAX_SCORE)
        if score >= 8000:
            available_memory_mb.append(12288.0)  # 12 GB
        elif score >= 2000:
            available_memory_mb.append(8192.0)   # 8 GB
        else:
            available_memory_mb.append(2048.0)   # 2 GB for IoT

    baseline_fwd_ms = []
    weight_sizes_bytes = []
    activation_sizes_bytes = []
    
    for i in range(total_layers):
        fwd_time = 10.0 if i not in (0, total_layers-1) else 12.0
        baseline_fwd_ms.append(fwd_time)
        weight_sizes_bytes.append(50 * 1024 * 1024) 
        activation_sizes_bytes.append(128 * 768 * 4) 

    config = {
        "num_layers": total_layers,
        "num_devices": num_devices,
        "simulated_hardware": selected_names,
        "computing_capacities": computing_capacities,
        "available_memory_mb": available_memory_mb,
        "node_bandwidths_mbps": selected_bandwidths,
        "baseline_fwd_ms": baseline_fwd_ms,
        "weight_sizes_bytes": weight_sizes_bytes,
        "activation_sizes_bytes": activation_sizes_bytes,
        "global_batch_size": 32,
        "micro_batch_size": 4,
        "num_stages": num_stages 
    }

    with open('asteroid_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ Generated asteroid_config.json:")
    print(f"   - Layers: {total_layers}")
    print(f"   - Devices: {num_devices} (Distribution: {distribution.upper()})")
    print(f"   - Pipeline Stages: {num_stages}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Asteroid Hybrid Pipeline Parallelism Configuration")
    
    parser.add_argument("--layers", type=int, default=12, 
                        help="Total number of transformer layers in the model")
    parser.add_argument("--min-devices", type=int, default=8, 
                        help="Minimum number of devices in the cluster")
    parser.add_argument("--max-devices", type=int, default=16, 
                        help="Maximum number of devices in the cluster")
    parser.add_argument("--stages", type=int, default=4, 
                        help="Target number of pipeline stages (P)")
    
    # --- ADDED: The distribution flag ---
    parser.add_argument("--distribution", type=str, choices=['random', 'balanced', 'skewed'], default='random',
                        help="Choose 'balanced' (all fast devices), 'skewed' (1 flagship + many IoT), or 'random'.")

    args = parser.parse_args()

    generate_asteroid_config(
        total_layers=args.layers,
        min_devices=args.min_devices,
        max_devices=args.max_devices,
        num_stages=args.stages,
        distribution=args.distribution
    )