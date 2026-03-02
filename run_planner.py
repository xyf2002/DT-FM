#!/usr/bin/env python3
"""
Asteroid HPP Planner - Aggregate node profiles and generate hpp_plan.json.

This script:
1. Loads profile_*.json files from the profiles directory
2. Parses cluster.conf for node IP/NIC/rank mappings
3. Builds DeviceSpec and populates AsteroidProfiler
4. Runs the DP planner to optimize stage assignment
5. Outputs hpp_plan.json with node_mapping for K8s deployment

Usage:
    python run_planner.py --profiles-dir ./profiles --output ./hpp_plan.json
    python run_planner.py --profiles-dir ./profiles --num-stages 3 --num-layers 12
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from asteroid.core.config import AsteroidConfig, DeviceSpec, HPPPlanConfig, NodeInfo
from asteroid.planner.profiler import AsteroidProfiler
from asteroid.planner.dp_planner import AsteroidPlanner


def parse_cluster_conf(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse cluster.conf to extract node information.
    
    Format: IP_ADDRESS  NIC_NAME  RANK  GPU_ID
    
    Returns:
        Dict mapping IP to {ip, nic, rank, gpu_id}
    """
    nodes = {}
    
    if not os.path.exists(path):
        print(f"Warning: cluster.conf not found at {path}")
        return nodes
    
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                ip = parts[0]
                nic = parts[1]
                try:
                    rank = int(parts[2])
                    gpu_id = int(parts[3])
                except ValueError:
                    print(f"Warning: Invalid rank/gpu_id on line {line_num}: {line}")
                    continue
                
                nodes[ip] = {
                    "ip": ip,
                    "nic": nic,
                    "rank": rank,
                    "gpu_id": gpu_id,
                }
            else:
                print(f"Warning: Malformed line {line_num} in cluster.conf: {line}")
    
    return nodes


def load_profiles(profiles_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load all profile_*.json files from directory.
    
    Returns:
        Dict mapping rank (int) to profile data
    """
    profiles = {}
    profiles_path = Path(profiles_dir)
    
    if not profiles_path.exists():
        print(f"Error: Profiles directory not found: {profiles_dir}")
        return profiles
    
    for profile_file in profiles_path.glob("profile_*.json"):
        try:
            with open(profile_file, "r") as f:
                data = json.load(f)
            
            # Extract rank from JSON field or filename (e.g., profile_xxx_rank0.json)
            rank = data.get("rank")
            if rank is None:
                # Try to parse rank from filename
                fname = profile_file.stem  # e.g., profile_ubuntu_rank0
                if "_rank" in fname:
                    try:
                        rank = int(fname.split("_rank")[-1])
                    except ValueError:
                        rank = None
            
            if rank is None:
                print(f"  Warning: No rank found in {profile_file.name}, skipping")
                continue
            
            rank = int(rank)
            hostname = data.get("hostname", "unknown")
            profiles[rank] = data
            print(f"  Loaded: {profile_file.name} (hostname: {hostname}, rank: {rank})")
        except json.JSONDecodeError as e:
            print(f"  Error loading {profile_file}: {e}")
    
    return profiles


def match_profile_to_cluster(
    profiles: Dict[int, Dict[str, Any]],
    cluster_nodes: Dict[str, Dict[str, Any]],
) -> Dict[int, Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Match profile ranks to cluster.conf entries.
    
    Returns:
        Dict mapping rank to (hostname, profile_data, cluster_info)
    """
    matched = {}
    
    # Try to match each cluster node by rank
    for ip, cluster_info in cluster_nodes.items():
        rank = cluster_info["rank"]
        
        # Direct match by rank
        if rank in profiles:
            profile_data = profiles[rank]
            hostname = profile_data.get("hostname", f"node-{rank}")
            matched[rank] = (hostname, profile_data, cluster_info)
            print(f"  Rank {rank}: {hostname} -> {ip}")
        else:
            print(f"  Warning: No profile match for rank {rank} ({ip})")
    
    return matched


def build_device_specs_and_profiler(
    matched_data: Dict[int, Tuple[str, Dict[str, Any], Dict[str, Any]]],
    config: AsteroidConfig,
) -> Tuple[List[DeviceSpec], AsteroidProfiler, Dict[int, NodeInfo]]:
    """
    Build DeviceSpec list, populate AsteroidProfiler, and create node_mapping.
    
    Returns:
        (device_specs, profiler, node_mapping)
    """
    device_specs = []
    node_mapping = {}
    
    # Create profiler with empty device list (we'll populate it)
    profiler = AsteroidProfiler(config, [])
    
    # IP to rank mapping for bandwidth
    ip_to_rank = {}
    profile_data_by_rank = {}
    
    for rank, (hostname, profile_data, cluster_info) in matched_data.items():
        hw = profile_data.get("hardware", {})
        
        # Build DeviceSpec
        device_spec = DeviceSpec(
            device_id=rank,
            device_type=hw.get("gpu_name", "unknown"),
            memory_budget_mb=hw.get("gpu_memory_mb", 4096),
            cuda_id=cluster_info["gpu_id"],
            compute_capacity=1.0,  # Could derive from profile times
        )
        device_specs.append(device_spec)
        profiler.devices[rank] = device_spec
        
        # Build NodeInfo for K8s deployment
        node_info = NodeInfo(
            hostname=hostname,
            ip=cluster_info["ip"],
            nic=cluster_info["nic"],
            gpu_id=cluster_info["gpu_id"],
            memory_mb=hw.get("gpu_memory_mb", 4096),
            architecture=hw.get("architecture", "x86_64"),
        )
        node_mapping[rank] = node_info
        
        # Store for bandwidth mapping
        ip_to_rank[cluster_info["ip"]] = rank
        profile_data_by_rank[rank] = profile_data
        
        # Load execution times into profiler
        profiler.exec_times[rank] = {}
        for layer_idx_str, bs_data in profile_data.get("exec_times", {}).items():
            layer_idx = int(layer_idx_str)
            profiler.exec_times[rank][layer_idx] = {}
            
            for bs_str, times in bs_data.items():
                bs = int(bs_str)
                # Skip OOM entries
                if isinstance(times.get("fwd_ms"), (int, float)):
                    profiler.exec_times[rank][layer_idx][bs] = (
                        float(times["fwd_ms"]),
                        float(times["bwd_ms"]),
                    )
    
    # Load bandwidth data
    for src_rank, profile_data in profile_data_by_rank.items():
        for target_ip, bw_mbps in profile_data.get("bandwidths_mbps", {}).items():
            if bw_mbps is not None and target_ip in ip_to_rank:
                dst_rank = ip_to_rank[target_ip]
                profiler.bandwidths[(src_rank, dst_rank)] = float(bw_mbps)
    
    # Set activation and weight sizes (approximate)
    if not profiler.activation_sizes:
        seq_len = config.max_seq_len
        d_model = config.embedding_dim
        profiler.activation_sizes = [seq_len * d_model * 4] * config.num_layers  # float32
        profiler.weight_sizes = [d_model * d_model * 4 * 4] * config.num_layers  # approx
    
    return device_specs, profiler, node_mapping


def create_synthetic_plan(
    num_devices: int,
    num_stages: int,
    num_layers: int,
    cluster_nodes: Dict[str, Dict[str, Any]],
) -> HPPPlanConfig:
    """
    Create a simple synthetic plan when no profiles available.
    Distributes devices evenly across stages.
    """
    print("\nCreating synthetic plan (no profiles)...")
    
    # Distribute devices across stages
    devices_per_stage = num_devices // num_stages
    remainder = num_devices % num_stages
    
    device_groups = {}
    current_device = 0
    for stage in range(num_stages):
        count = devices_per_stage + (1 if stage < remainder else 0)
        device_groups[stage] = list(range(current_device, current_device + count))
        current_device += count
    
    # Equal layer partition
    layers_per_stage = num_layers // num_stages
    partition_points = []
    for i in range(1, num_stages):
        partition_points.append(i * layers_per_stage)
    
    # Simple micro-batch allocation
    micro_batch_alloc = {}
    for stage, devices in device_groups.items():
        micro_batch_alloc[stage] = {d: 4 for d in devices}
    
    # Build node_mapping from cluster.conf
    node_mapping = {}
    for ip, info in cluster_nodes.items():
        rank = info["rank"]
        if rank < num_devices:
            node_mapping[rank] = NodeInfo(
                hostname=f"node-{rank}",
                ip=ip,
                nic=info["nic"],
                gpu_id=info["gpu_id"],
                memory_mb=4096,
            )
    
    return HPPPlanConfig(
        num_stages=num_stages,
        partition_points=partition_points,
        device_groups=device_groups,
        micro_batch_alloc=micro_batch_alloc,
        estimated_latency_ms=0.0,
        node_mapping=node_mapping,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Asteroid HPP Planner - Generate hpp_plan.json from profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_planner.py --profiles-dir ./profiles
  
  # Custom output path
  python run_planner.py --profiles-dir ./profiles --output ./deploy/hpp_plan.json
  
  # Custom stage count
  python run_planner.py --profiles-dir ./profiles --num-stages 3
        """
    )
    
    parser.add_argument(
        "--profiles-dir", type=str, default="./profiles",
        help="Directory containing profile_*.json files (default: ./profiles)"
    )
    parser.add_argument(
        "--cluster-conf", type=str, default="./cluster.conf",
        help="Path to cluster.conf (default: ./cluster.conf)"
    )
    parser.add_argument(
        "--output", type=str, default="./hpp_plan.json",
        help="Output path for hpp_plan.json (default: ./hpp_plan.json)"
    )
    parser.add_argument(
        "--num-stages", type=int, default=2,
        help="Number of pipeline stages (default: 2)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=12,
        help="Number of model layers (default: 12)"
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=4,
        help="Micro-batch size (default: 4)"
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=256,
        help="Global batch size (default: 256)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic plan without running DP planner"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ASTEROID HPP PLANNER")
    print("=" * 60)
    
    # Load cluster configuration
    print(f"\nLoading cluster configuration from {args.cluster_conf}")
    cluster_nodes = parse_cluster_conf(args.cluster_conf)
    print(f"  Found {len(cluster_nodes)} nodes in cluster.conf")
    
    # Load profiles
    print(f"\nLoading profiles from {args.profiles_dir}")
    profiles = load_profiles(args.profiles_dir)
    print(f"  Loaded {len(profiles)} profile files")
    
    # Determine world size
    world_size = max(len(cluster_nodes), len(profiles), 1)
    
    # Handle case with no profiles
    if not profiles:
        if args.synthetic or not cluster_nodes:
            print("\nNo profiles found, generating synthetic plan...")
            plan = create_synthetic_plan(
                num_devices=world_size,
                num_stages=args.num_stages,
                num_layers=args.num_layers,
                cluster_nodes=cluster_nodes,
            )
        else:
            print("Error: No profiles found. Run profile_and_gather.yaml first.")
            print("  Or use --synthetic flag to generate a basic plan.")
            return 1
    else:
        # Match profiles to cluster nodes
        print("\nMatching profiles to cluster nodes...")
        matched_data = match_profile_to_cluster(profiles, cluster_nodes)
        
        if not matched_data:
            print("Error: Could not match any profiles to cluster nodes.")
            return 1
        
        world_size = len(matched_data)
        
        # Build config
        config = AsteroidConfig(
            num_layers=args.num_layers,
            world_size=world_size,
            num_stages=args.num_stages,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
        )
        
        # Build profiler data
        print("\nBuilding profiler data...")
        device_specs, profiler, node_mapping = build_device_specs_and_profiler(
            matched_data, config
        )
        print(f"  Device specs: {len(device_specs)}")
        print(f"  Exec times loaded: {sum(len(l) for d in profiler.exec_times.values() for l in d.values())}")
        print(f"  Bandwidth measurements: {len(profiler.bandwidths)}")
        
        # Run planner
        print(f"\nRunning DP planner (stages={args.num_stages}, layers={args.num_layers})...")
        
        try:
            planner = AsteroidPlanner(
                profiler=profiler,
                cfg=config,
                devices=device_specs,
            )
            
            plan = planner.plan()
            
            # Attach node mapping
            plan.node_mapping = node_mapping
            
        except Exception as e:
            print(f"Warning: DP planner failed ({e}), using fallback plan")
            plan = create_synthetic_plan(
                num_devices=world_size,
                num_stages=args.num_stages,
                num_layers=args.num_layers,
                cluster_nodes=cluster_nodes,
            )
            # Merge node_mapping
            plan.node_mapping.update(node_mapping)
    
    # Save plan
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(plan.to_json(), f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("HPP PLAN GENERATED")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Stages: {plan.num_stages}")
    print(f"Partition points: {plan.partition_points}")
    print(f"Device groups:")
    for stage, devices in plan.device_groups.items():
        print(f"  Stage {stage}: devices {devices}")
    print(f"Estimated latency: {plan.estimated_latency_ms:.2f} ms")
    print(f"World size: {sum(len(d) for d in plan.device_groups.values())}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
