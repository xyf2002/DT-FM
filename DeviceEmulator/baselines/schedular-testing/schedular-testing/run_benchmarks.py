import subprocess
import argparse
import os
from datetime import datetime

def run_command(command, log_file):
    """Executes a terminal command, prints the output, and writes it to the log."""
    print(f"Running: {' '.join(command)}")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"COMMAND: {' '.join(command)}\n")
    log_file.write(f"{'='*80}\n")
    
    try:
        # capture_output=True grabs stdout and stderr seamlessly
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Print to terminal so you can watch it live
        print(result.stdout)
        
        # Write to our master results file
        log_file.write(result.stdout)
        
        if result.stderr:
            log_file.write("\n--- STDERR ---\n")
            log_file.write(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command. Exit code: {e.returncode}")
        print(e.stderr)
        log_file.write(f"\n❌ ERROR: Command failed with exit code {e.returncode}\n")
        log_file.write(e.stdout)
        log_file.write(e.stderr)

def main():
    parser = argparse.ArgumentParser(description="Automated Benchmarking Pipeline for Edge Schedulers")
    parser.add_argument("--layers", type=int, default=32, help="Number of transformer layers")
    parser.add_argument("--pp-size", type=int, default=4, help="Pipeline stages (P)")
    parser.add_argument("--dp-size", type=int, default=4, help="Data parallel replicas (Used by DT-FM to set total N)")
    parser.add_argument("--distribution", type=str, choices=['random', 'balanced', 'skewed'], default='random')
    
    # --- ADDED: Batching Parameters ---
    parser.add_argument("--global-batch-size", type=int, default=32, help="Total global batch size")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Micro-batch chunk size")
    
    parser.add_argument("--output", type=str, default="benchmark_results.txt", help="Output file for the logs")
    args = parser.parse_args()

    # 1. Construct the configuration generator command including batch sizes
    config_cmd = [
        "python3", "generate_unified_configs.py",
        "--layers", str(args.layers),
        "--pp-size", str(args.pp_size),
        "--dp-size", str(args.dp_size),
        "--distribution", args.distribution,
        "--global-batch-size", str(args.global_batch_size),
        "--micro-batch-size", str(args.micro_batch_size)
    ]

    # 2. Define the exact paths to your three schedulers
    schedulers = [
        ("Confidant", ["python3", "confident/confident-schedular-test.py"]),
        ("Asteroid", ["python3", "astroid/simulate_asteroid_scheduler.py"]),
        ("DT-FM", ["python3", "dtfm/simulate_dtfm_scheduler.py"])
    ]

    # Open the master log file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.output, 'w') as log_file:
        log_file.write(f"Scheduler Comparison Benchmark Run - {timestamp}\n")
        log_file.write(f"Parameters: Layers={args.layers}, PP={args.pp_size}, DP={args.dp_size}, Dist={args.distribution}\n")
        log_file.write(f"Batching: Global={args.global_batch_size}, Micro={args.micro_batch_size}\n")
        
        # Step 1: Generate the common truth
        print("\n--- Step 1: Generating Unified Configs ---")
        run_command(config_cmd, log_file)
        
        # Step 2: Race the schedulers
        print("\n--- Step 2: Running Schedulers ---")
        for name, cmd in schedulers:
            print(f"\n🚀 Evaluating {name}...")
            run_command(cmd, log_file)

    print(f"\n✅ All runs completed! Master log saved to: {args.output}")

if __name__ == "__main__":
    main()