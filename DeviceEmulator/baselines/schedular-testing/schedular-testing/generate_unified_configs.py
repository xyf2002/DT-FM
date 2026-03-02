import json
import random
import os
import pickle
import argparse
import numpy as np

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

def generate_unified_configs(total_layers, pp_size, dp_size, distribution, global_batch_size, micro_batch_size):
    # To satisfy DT-FM's strict grid, total devices must be PP * DP
    num_devices = pp_size * dp_size
    MAX_SCORE = HARDWARE_POOL["Snapdragon 8 Elite Gen 5"]
    
    print(f"\n⚙️  Generating Unified Topology for {num_devices} Devices ({distribution.upper()})...")
    print(f"   - Global Batch Size: {global_batch_size}, Micro-Batch Size: {micro_batch_size}")
    
    # =========================================================================
    # 1. CORE SAMPLING (Done ONCE for all schedulers)
    # =========================================================================
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
    available_memory_gb = []
    available_memory_mb = []
    
    for base_name in base_device_types:
        score = HARDWARE_POOL[base_name]
        computing_capacities.append(score / MAX_SCORE)
        if score >= 8000:
            available_memory_gb.append(12.0)
            available_memory_mb.append(12288.0)
        elif score >= 2000:
            available_memory_gb.append(8.0)
            available_memory_mb.append(8192.0)
        else:
            available_memory_gb.append(2.0)
            available_memory_mb.append(2048.0)

    # =========================================================================
    # 2. MODEL PROFILING ESTIMATIONS (GPT-2 Style)
    # =========================================================================
    baseline_fwd_ms = []
    baseline_bwd_ms = []
    confident_layer_profiles = []
    
    d_model = 768
    max_seq_len = 128
    
    for i in range(total_layers):
        fwd_time = 10.0 if i not in (0, total_layers-1) else 12.0
        bwd_time = fwd_time * 2.0
        baseline_fwd_ms.append(fwd_time)
        baseline_bwd_ms.append(bwd_time)
        
        # Confidant schema
        confident_layer_profiles.append({
            "layer_idx": i,
            "forward_ms": fwd_time,
            "backward_ms": bwd_time,
            "memory_mb": 1500.0  # Baseline static memory per layer
        })
        
    # Sizes
    params_per_layer = 12 * d_model * d_model
    total_params = params_per_layer * total_layers
    
    # Bytes
    weight_sizes_bytes = [params_per_layer * 4] * total_layers
    activation_sizes_bytes = [max_seq_len * d_model * 4] * total_layers
    
    # --- DYNAMIC MICRO-BATCH MATH ---
    # Megabytes (For Confidant & DT-FM Partitioner)
    output_sizes_mb = [(micro_batch_size * max_seq_len * d_model * 4) / (1024 * 1024)] * total_layers
    
    # Gigabytes (For DT-FM GCMA AllReduce math)
    send_gradient_size_gb = (total_params * 4 / pp_size) / (1024 ** 3)
    send_activation_size_gb = (micro_batch_size * max_seq_len * d_model * 4) / (1024 ** 3)

    # 2D Network Matrices (For DT-FM)
    peer_bandwidth_gbps = np.zeros((num_devices, num_devices))
    peer_delay_ms = np.zeros((num_devices, num_devices))
    for i in range(num_devices):
        for j in range(num_devices):
            if i == j:
                peer_bandwidth_gbps[i, j] = 1000.0
                peer_delay_ms[i, j] = 0.001
            else:
                bottleneck_mbs = min(node_bandwidths_mbs[i], node_bandwidths_mbs[j])
                peer_bandwidth_gbps[i, j] = bottleneck_mbs * 8 / 1000.0
                peer_delay_ms[i, j] = random.uniform(5.0, 20.0) if bottleneck_mbs < 10 else random.uniform(1.0, 5.0)


    # =========================================================================
    # 3. GENERATE CONFIDANT CONFIG
    # =========================================================================
    confidant_config = {
        "num_layers": total_layers,
        "num_devices": num_devices,
        "simulated_hardware": selected_names,
        "computing_capacities": computing_capacities,
        "bandwidths": node_bandwidths_mbs,
        "available_memory": available_memory_gb,
        "output_sizes": output_sizes_mb,
        "layer_profiles": confident_layer_profiles
    }
    with open('confident_config.json', 'w') as f:
        json.dump(confidant_config, f, indent=4)

    # =========================================================================
    # 4. GENERATE ASTEROID CONFIG
    # =========================================================================
    asteroid_config = {
        "num_layers": total_layers,
        "num_devices": num_devices,
        "simulated_hardware": selected_names,
        "computing_capacities": computing_capacities,
        "available_memory_mb": available_memory_mb,
        "node_bandwidths_mbps": node_bandwidths_mbs,
        "baseline_fwd_ms": baseline_fwd_ms,
        "weight_sizes_bytes": weight_sizes_bytes,
        "activation_sizes_bytes": activation_sizes_bytes,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "num_stages": pp_size # Asteroid's 'P' parameter
    }
    with open('asteroid_config.json', 'w') as f:
        json.dump(asteroid_config, f, indent=4)

    # =========================================================================
    # 5. GENERATE DT-FM CONFIG
    # =========================================================================
    dtfm_config = {
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
        json.dump(dtfm_config, f, indent=4)

    print(f"✅ Successfully wrote 3 aligned configuration files!")
    print(f"   -> confident_config.json")
    print(f"   -> asteroid_config.json")
    print(f"   -> dtfm_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Unified Configs for Schedulers")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--pp-size", type=int, default=4, help="Pipeline stages (P)")
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel replicas (Used by DT-FM to set total N)")
    parser.add_argument("--distribution", type=str, choices=['random', 'balanced', 'skewed'], default='random')
    
    # --- ADDED: Batching Arguments ---
    parser.add_argument("--global-batch-size", type=int, default=32, help="Total global batch size for training")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Size of individual micro-batch chunks")
    
    args = parser.parse_args()

    generate_unified_configs(args.layers, args.pp_size, args.dp_size, args.distribution, args.global_batch_size, args.micro_batch_size)