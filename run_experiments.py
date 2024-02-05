"""
    runs `main.py` for different datasets and/or features in parallel within a multi-GPU environment.
    note: need to generate dataset first, run `./prepare_data.py`
"""
import os
import concurrent.futures

from main import main as run_experiment

# Environment variables
gpu_ids = [0, 1, 2, 3] # List of GPU IDs to use
job_per_gpu = 2        # Number of jobs to run on each GPU

# experiment variables
dataset_names = ["IHAMPS1", "IMONTR17", "IMONTR172", "IQUEBECM21", "IWESTM40"] 
feature_names = ["Temperature_C", "Humidity_%", "Pressure_hPa"]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    n_workers = len(gpu_ids) * job_per_gpu

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        i = 0
        for dataset in dataset_names:
            for feat in feature_names:
                # Set up the experiment parameters
                train_path = f"dataset/synth_ts_data/train_{dataset}_{feat}.csv"
                test_path = f"dataset/synth_ts_data/test_{dataset}_{feat}.csv"
                output_file = f"results/{dataset}_{feat}.txt"
                gpu_id = gpu_ids[i % len(gpu_ids)] # Assign GPUs in a round-robin fashion
                i += 1

                # Submit an experiment for execution and store the future object
                futures[executor.submit(run_experiment, train_path, test_path, output_file, gpu_id)] = (train_path, test_path, output_file, gpu_id)

        # Iterate over completed futures as they become available
        for completed_future in concurrent.futures.as_completed(futures):
            train_path, test_path, output_file, gpu_id = futures[completed_future]
            try:
                completed_future.result() # Blocked until result is ready
            except Exception as e:
                print(f"Training failed for dataset: {train_path} with error: {e}")

        print("Multi-processes finished")

