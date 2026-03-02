#!/usr/bin/env python3
"""
Asteroid K8s Manifest Generator - Generate Kubernetes manifests from hpp_plan.json.

Renders Jinja2 templates to create:
- Headless Service for Rank 0 DNS
- ConfigMap with hpp_plan.json
- Job manifests for each worker rank

Usage:
    python generate_manifests.py --plan ./hpp_plan.json
    python generate_manifests.py --plan ./hpp_plan.json --image myregistry/asteroid:v1.0.0
    python generate_manifests.py --plan ./hpp_plan.json --apply
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
except ImportError:
    print("Error: Jinja2 not installed. Run: pip install jinja2")
    sys.exit(1)


def load_plan(path: str) -> Dict[str, Any]:
    """Load hpp_plan.json file."""
    with open(path, "r") as f:
        return json.load(f)


def resolve_k8s_node_names() -> Dict[str, str]:
    """Query kubectl to build a map of IP -> K8s node name.
    
    This handles cases where the K8s node name differs from the OS hostname
    (e.g. when K3S_NODE_NAME is set to resolve hostname collisions).
    Returns empty dict if kubectl is unavailable.
    """
    try:
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o",
             "jsonpath={range .items[*]}{.status.addresses[?(@.type==\"InternalIP\")].address}={.metadata.name}{\"\\n\"}{end}"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "KUBECONFIG": os.path.expanduser("~/.kube/config")},
        )
        if result.returncode == 0:
            mapping = {}
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    ip, name = line.split("=", 1)
                    mapping[ip.strip()] = name.strip()
            return mapping
    except Exception:
        pass
    return {}


def setup_jinja_env(templates_dir: str) -> Environment:
    """Configure Jinja2 environment."""
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env


def generate_configmap(
    env: Environment,
    plan: Dict[str, Any],
    namespace: str,
) -> str:
    """Generate ConfigMap YAML containing hpp_plan.json."""
    try:
        template = env.get_template("configmap.yaml.j2")
    except TemplateNotFound:
        print("Error: configmap.yaml.j2 template not found")
        sys.exit(1)
    
    return template.render(
        namespace=namespace,
        hpp_plan_json=json.dumps(plan, indent=2),
    )


def generate_headless_service(
    env: Environment,
    namespace: str,
    master_port: int,
) -> str:
    """Generate Headless Service YAML for Rank 0 DNS."""
    try:
        template = env.get_template("headless_service.yaml.j2")
    except TemplateNotFound:
        print("Error: headless_service.yaml.j2 template not found")
        sys.exit(1)
    
    return template.render(
        namespace=namespace,
        master_port=master_port,
    )


def generate_job_manifest(
    env: Environment,
    plan: Dict[str, Any],
    rank: int,
    stage: int,
    namespace: str,
    image: str,
    master_port: int,
    world_size: int,
    k8s_node_map: Dict[str, str] = None,
    extra_vars: Dict[str, Any] = None,
) -> str:
    """Generate Job YAML for a specific rank."""
    try:
        template = env.get_template("stage_job.yaml.j2")
    except TemplateNotFound:
        print("Error: stage_job.yaml.j2 template not found")
        sys.exit(1)
    
    # Get node info from plan
    node_mapping = plan.get("node_mapping", {})
    node_info = node_mapping.get(str(rank), {})
    
    # Resolve K8s node name: prefer IP-based lookup over hostname
    node_hostname = node_info.get("hostname", f"node-{rank}")
    node_ip = node_info.get("ip", "")
    if k8s_node_map and node_ip in k8s_node_map:
        node_hostname = k8s_node_map[node_ip]
    
    # Get micro-batch allocation
    stage_alloc = plan.get("micro_batch_alloc", {}).get(str(stage), {})
    micro_batch_size = stage_alloc.get(str(rank), 4)
    
    # Template variables
    vars = {
        "rank": rank,
        "stage": stage,
        "world_size": world_size,
        "namespace": namespace,
        "master_port": master_port,
        "image": image,
        "node_hostname": node_hostname,
        "memory_mb": node_info.get("memory_mb", 4096),
        "gpu_id": node_info.get("gpu_id", 0),
        "nccl_ifname": "eth0",  # Always eth0 inside K8s pods
        "micro_batch_size": micro_batch_size,
        "is_last_stage": (stage == plan.get("num_stages", 1) - 1),
    }
    
    # Merge extra variables
    if extra_vars:
        vars.update(extra_vars)
    
    return template.render(**vars)


def write_manifest(path: Path, content: str, dry_run: bool = False) -> None:
    """Write manifest to file."""
    if dry_run:
        print(f"[DRY RUN] Would write: {path}")
    else:
        path.write_text(content)
        print(f"[CREATED] {path}")


def generate_apply_script(output_dir: Path, namespace: str) -> str:
    """Generate shell script to apply all manifests."""
    return f"""#!/bin/bash
# Apply all Asteroid K8s manifests
# Generated by generate_manifests.py

set -e

MANIFEST_DIR="{output_dir}"
NAMESPACE="{namespace}"

echo "Applying Asteroid manifests from $MANIFEST_DIR"
echo "Namespace: $NAMESPACE"
echo ""

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

# Apply manifests in order
echo "Applying ConfigMap..."
kubectl apply -f "$MANIFEST_DIR/00-configmap.yaml"

echo "Applying Headless Service..."
kubectl apply -f "$MANIFEST_DIR/01-headless-service.yaml"

echo "Applying Worker Jobs..."
for job in "$MANIFEST_DIR"/02-job-rank-*.yaml; do
    echo "  Applying $(basename $job)..."
    kubectl apply -f "$job"
done

echo ""
echo "All manifests applied successfully!"
echo ""

# Wait for pods to start
echo "Waiting for pods to start..."
kubectl wait --for=condition=Ready pod -l app=asteroid -n "$NAMESPACE" --timeout=300s || true

# Show status
echo ""
echo "Pod Status:"
kubectl get pods -l app=asteroid -n "$NAMESPACE" -o wide
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes manifests from hpp_plan.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate manifests
  python generate_manifests.py --plan ./hpp_plan.json

  # Custom image and namespace
  python generate_manifests.py --plan ./hpp_plan.json \\
      --image myregistry/asteroid:v1.0.0 \\
      --namespace ml-training

  # Dry run (show what would be generated)
  python generate_manifests.py --plan ./hpp_plan.json --dry-run

  # Generate and apply
  python generate_manifests.py --plan ./hpp_plan.json --apply
        """
    )
    
    parser.add_argument(
        "--plan", type=str, required=True,
        help="Path to hpp_plan.json"
    )
    parser.add_argument(
        "--templates-dir", type=str, default=None,
        help="Jinja2 templates directory (default: deploy/templates)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for manifests (default: deploy/generated)"
    )
    parser.add_argument(
        "--image", type=str, default="asteroid:latest",
        help="Docker image for worker pods (default: asteroid:latest)"
    )
    parser.add_argument(
        "--namespace", type=str, default="default",
        help="Kubernetes namespace (default: default)"
    )
    parser.add_argument(
        "--master-port", type=int, default=29500,
        help="PyTorch distributed port (default: 29500)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without writing files"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply manifests using kubectl after generation"
    )
    parser.add_argument(
        "--image-pull-policy", type=str, default="IfNotPresent",
        choices=["Always", "IfNotPresent", "Never"],
        help="Image pull policy (default: IfNotPresent)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    templates_dir = Path(args.templates_dir) if args.templates_dir else script_dir / "templates"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "generated"
    plan_path = Path(args.plan)
    
    # Validate inputs
    if not plan_path.exists():
        print(f"Error: Plan file not found: {plan_path}")
        return 1
    
    if not templates_dir.exists():
        print(f"Error: Templates directory not found: {templates_dir}")
        return 1
    
    print("=" * 60)
    print("ASTEROID K8S MANIFEST GENERATOR")
    print("=" * 60)
    print(f"Plan: {plan_path}")
    print(f"Templates: {templates_dir}")
    print(f"Output: {output_dir}")
    print(f"Image: {args.image}")
    print(f"Namespace: {args.namespace}")
    
    # Load plan
    print(f"\nLoading plan from {plan_path}...")
    plan = load_plan(plan_path)
    
    world_size = plan.get("world_size", 1)
    num_stages = plan.get("num_stages", 1)
    
    print(f"  World size: {world_size}")
    print(f"  Stages: {num_stages}")
    
    # Setup Jinja2
    env = setup_jinja_env(str(templates_dir))
    
    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    extra_vars = {
        "image_pull_policy": args.image_pull_policy,
    }
    
    # 1. Generate ConfigMap
    print("\nGenerating ConfigMap...")
    configmap_content = generate_configmap(env, plan, args.namespace)
    configmap_path = output_dir / "00-configmap.yaml"
    write_manifest(configmap_path, configmap_content, args.dry_run)
    generated_files.append(("ConfigMap", configmap_path))
    
    # 2. Generate Headless Service
    print("Generating Headless Service...")
    service_content = generate_headless_service(env, args.namespace, args.master_port)
    service_path = output_dir / "01-headless-service.yaml"
    write_manifest(service_path, service_content, args.dry_run)
    generated_files.append(("Service", service_path))
    
    # 3. Generate Jobs for each rank
    print("Generating Worker Jobs...")
    device_groups = plan.get("device_groups", {})
    
    # Resolve K8s node names from IPs (handles hostname != K8s node name)
    k8s_node_map = resolve_k8s_node_names()
    if k8s_node_map:
        print(f"  Resolved {len(k8s_node_map)} K8s node name(s) from cluster")
    
    for stage_str, devices in device_groups.items():
        stage = int(stage_str)
        for rank in devices:
            job_content = generate_job_manifest(
                env=env,
                plan=plan,
                rank=rank,
                stage=stage,
                namespace=args.namespace,
                image=args.image,
                master_port=args.master_port,
                world_size=world_size,
                k8s_node_map=k8s_node_map,
                extra_vars=extra_vars,
            )
            job_path = output_dir / f"02-job-rank-{rank}.yaml"
            write_manifest(job_path, job_content, args.dry_run)
            generated_files.append((f"Job (Rank {rank}, Stage {stage})", job_path))
    
    # 4. Generate apply script
    if not args.dry_run:
        apply_script_content = generate_apply_script(output_dir, args.namespace)
        apply_script_path = output_dir / "apply.sh"
        apply_script_path.write_text(apply_script_content)
        apply_script_path.chmod(0o755)
        generated_files.append(("Apply Script", apply_script_path))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Generated {len(generated_files)} files:")
    for name, path in generated_files:
        print(f"  - {name}: {path.name}")
    print("=" * 60)
    
    if not args.dry_run:
        print(f"\nTo apply manually:")
        print(f"  kubectl apply -f {output_dir}/")
        print(f"\nOr use the apply script:")
        print(f"  bash {output_dir}/apply.sh")
    
    # Apply if requested
    if args.apply and not args.dry_run:
        print("\nApplying manifests...")
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", str(output_dir) + "/"],
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return 1
            
            print("\nPod status:")
            subprocess.run([
                "kubectl", "get", "pods",
                "-l", "app=asteroid",
                "-n", args.namespace,
                "-o", "wide"
            ])
        except FileNotFoundError:
            print("Error: kubectl not found in PATH")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
