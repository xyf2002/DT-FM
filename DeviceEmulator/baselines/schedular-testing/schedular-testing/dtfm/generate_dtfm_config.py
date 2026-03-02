import json
import random
import os
import pickle
import argparse
import numpy as np

HARDWARE_POOL = {
    "Snapdragon 8 Elite Gen 5": 16226.0,
    "Snapdragon 8 Gen 3": 7113.0,
    "Apple A18 Pro": 3790.0,
    "Snapdragon 680 (IoT)": 95.0,
    "Nvidia Tegra X1 (IoT)": 60.0,
    "Rockchip RK3588 (IoT)": 240.0
}

def parse_fedscale_network_trace(filepath="client_device_capacity"):
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

def generate_dtfm_config(total_layers, pp_size, dp_size, distribution):
    num_devices = pp_size * dp_size
    MAX_SCORE = HARDWARE_POOL["Snapdragon 8 Elite Gen 5"]
    
    # 1. Distribution Logic
    if distribution == "balanced":
        top_tier = ["Snapdragon 8 Elite Gen 5", "Snapdragon 8 Gen 3", "Apple A18 Pro"]
        base_device_types = random.choices(top_tier, k=num_devices)
    elif distribution == "skewed":
        base_device_types = ["Snapdragon 8 Elite Gen 5"]
        iot_tier = ["Snapdragon 680 (IoT)", "Nvidia Tegra X1 (IoT)", "Rockchip RK3588 (IoT)"]
        base_device_types += random.choices(iot_tier, k=num_devices - 1)
        random.shuffle(base_device_types)
    else:
        base_device_types = random.choices(list(HARDWARE_POOL.keys()), k=num_devices)
    
    selected_names = [f"{name} (Node {i})" for i, name in enumerate(base_device_types)]
    
    network_pool = parse_fedscale_network_trace("client_device_capacity")
    node_bandwidths_mbs = random.choices(network_pool, k=num_devices) 
    
    computing_capacities = []
    
    for base_name in base_device_types:
        score = HARDWARE_POOL[base_name]
        computing_capacities.append(score / MAX_SCORE)

    # 2. Build All-Pairs Network Matrices (Gbps and ms) for GCMA
    peer_bandwidth_gbps = np.zeros((num_devices, num_devices))
    peer_delay_ms = np.zeros((num_devices, num_devices))
    
    for i in range(num_devices):
        for j in range(num_devices):
            if i == j:
                peer_bandwidth_gbps[i, j] = 1000.0 # Internal memory speed
                peer_delay_ms[i, j] = 0.001
            else:
                # Link is limited by the slower of the two endpoints
                bottleneck_mbs = min(node_bandwidths_mbs[i], node_bandwidths_mbs[j])
                peer_bandwidth_gbps[i, j] = bottleneck_mbs * 8 / 1000.0 # Convert MB/s to Gbps
                # Simulated ping/delay based on bandwidth tier
                peer_delay_ms[i, j] = random.uniform(5.0, 20.0) if bottleneck_mbs < 10 else random.uniform(1.0, 5.0)

    # 3. Layer Profiles & Data Sizes
    baseline_fwd_ms = []
    baseline_bwd_ms = []
    
    # GPT-2 approximations
    d_model = 768
    micro_batch_size = 4
    max_seq_len = 128
    
    for i in range(total_layers):
        fwd_time = 10.0 if i not in (0, total_layers-1) else 12.0
        baseline_fwd_ms.append(fwd_time)
        baseline_bwd_ms.append(fwd_time * 2.0)
        
    params_per_layer = 12 * d_model * d_model
    total_params = params_per_layer * total_layers
    
    # GCMA uses float32 GB sizing
    send_gradient_size_gb = (total_params * 4 / pp_size) / (1024 ** 3)
    send_activation_size_gb = (micro_batch_size * max_seq_len * d_model * 4) / (1024 ** 3)
    
    # We output a flat activation size per layer (MB) for the DP partitioner
    output_sizes_mb = [(micro_batch_size * max_seq_len * d_model * 4) / (1024 * 1024)] * total_layers

    config = {
        "num_layers": total_layers,
        "num_devices": num_devices,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "simulated_hardware": selected_names,
        "computing_capacities": computing_capacities,
        "peer_bandwidth_gbps": peer_bandwidth_gbps.tolist(),
        "peer_delay_ms": peer_delay_ms.tolist(),
        "send_gradient_size_gb": send_gradient_size_gb,
        "send_activation_size_gb": send_activation_size_gb,
        "baseline_fwd_ms": baseline_fwd_ms,
        "baseline_bwd_ms": baseline_bwd_ms,
        "output_sizes_mb": output_sizes_mb
    }

    with open('dtfm_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ Generated dtfm_config.json:")
    print(f"   - Layers: {total_layers}")
    print(f"   - Topology: {pp_size} Pipeline Stages x {dp_size} DP Replicas = {num_devices} Devices")
    print(f"   - Distribution: {distribution.upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DT-FM GCMA Configuration")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--pp-size", type=int, default=4, help="Number of pipeline stages")
    parser.add_argument("--dp-size", type=int, default=2, help="Number of data parallel replicas")
    parser.add_argument("--distribution", type=str, choices=['random', 'balanced', 'skewed'], default='random')
    args = parser.parse_args()

    generate_dtfm_config(args.layers, args.pp_size, args.dp_size, args.distribution)