import json
import random

# ==============================================================================
# 1. Expanded Hardware Pool (Sourced from your AI-Benchmark table)
# ==============================================================================
# A diverse mix of Flagships, Mid-range, and IoT/Older edge devices.
HARDWARE_POOL = {
    "Snapdragon 8 Elite Gen 5": 16226.0,
    "Dimensity 9500": 15015.0,
    "Exynos 2400": 10652.0,
    "Dimensity 9400": 8074.0,
    "Snapdragon 8 Gen 3": 7113.0,
    "Snapdragon 8 Gen 2": 5238.0,
    "Google Tensor G4": 3895.0,
    "Apple A18 Pro": 3790.0,
    "Apple A17 Pro": 3428.0,
    "Google Tensor G3": 2829.0,
    "Snapdragon 888": 1724.0,
    "Exynos 2100": 495.0,
    "Kirin 9000 5G": 452.0,
    "Rockchip RK3588 (IoT)": 240.0,
    "Snapdragon 680": 95.0,
    "Nvidia Tegra X1 (IoT)": 60.0
}

def generate_expanded_config(num_devices=4, total_layers=12):
    # The absolute fastest chip is our 1.0 baseline
    MAX_SCORE = HARDWARE_POOL["Snapdragon 8 Elite Gen 5"]

    # 1. Randomly sample 'num_devices' from our pool for this simulation run
    selected_names = random.sample(list(HARDWARE_POOL.keys()), num_devices)
    
    computing_capacities = []
    available_memory = []
    bandwidths = []

    # 2. Dynamically assign realistic constraints based on the device's compute tier
    for name in selected_names:
        score = HARDWARE_POOL[name]
        
        # Capacity ratio
        computing_capacities.append(score / MAX_SCORE)
        
        # RAM constraint (GB) and Network Tier (MB/s) heuristics
        if score >= 8000:
            available_memory.append(12.0)    # Flagship: 12GB RAM
            bandwidths.append(50.0)          # Strong 5G / Wi-Fi 6
        elif score >= 2000:
            available_memory.append(8.0)     # High/Mid: 8GB RAM
            bandwidths.append(25.0)          # Standard Wi-Fi
        elif score >= 300:
            available_memory.append(4.0)     # Mid/Low: 4GB RAM
            bandwidths.append(10.0)          # Average 4G
        else:
            available_memory.append(2.0)     # IoT/Old: 2GB RAM
            bandwidths.append(3.0)           # Congested / Poor connection

    # ==============================================================================
    # 3. Heuristic Layer Profiling Formula
    # ==============================================================================
    baseline_total_inference_ms = 120.0 
    base_fwd_ms = baseline_total_inference_ms / total_layers

    layer_profiles = []
    for i in range(total_layers):
        layer_fwd = base_fwd_ms
        layer_mem = 1500.0 # Baseline 1.5GB
        
        if i == 0:
            layer_fwd *= 1.15
            layer_mem += 200.0
        elif i == total_layers - 1:
            layer_fwd *= 1.20
            layer_mem += 300.0
        else:
            noise = random.uniform(0.97, 1.03)
            layer_fwd *= noise

        layer_bwd = layer_fwd * 2.0

        layer_profiles.append({
            "layer_idx": i,
            "forward_ms": round(layer_fwd, 2),
            "backward_ms": round(layer_bwd, 2),
            "memory_mb": round(layer_mem, 2)
        })

    # ==============================================================================
    # 4. Generate JSON
    # ==============================================================================
    config = {
        "num_layers": total_layers,
        "num_devices": num_devices,
        "simulated_hardware": selected_names,
        "computing_capacities": computing_capacities,
        "bandwidths": bandwidths,
        "available_memory": available_memory,
        "output_sizes": [3.0] * total_layers, 
        "layer_profiles": layer_profiles
    }

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ Generated config.json with {num_devices} random edge devices!")
    for i in range(num_devices):
        print(f"  Device {i}: {selected_names[i]:<25} | "
              f"Capacity: {computing_capacities[i]:.4f} | "
              f"RAM: {available_memory[i]:>4}GB | "
              f"Net: {bandwidths[i]:>4} MB/s")

if __name__ == "__main__":
    # You can change the number of simulated clients here
    generate_expanded_config(num_devices=8, total_layers=12)