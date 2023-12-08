"""
    runs `main.py` for different datasets and/or features in parallel within a multi-GPU environment.
    note: need to generate dataset first, run `./prepare_data.py`
"""
import os
import multiprocessing as mp
from main import main

# environment variables
gpu_ids = [0, 1, 2, 3]
nbr_workers = len(gpu_ids)

# experiment variables
dataset_names = ["IHAMPS1", "IMONTR17", "IMONTR172", "IQUEBECM21", "IWESTM40"] 
feature_names = ["Temperature_C", "Humidity_%", "Pressure_hPa"]

if __name__ == "__main__":
    experiment_args = []
    i = 0
    for dataset in dataset_names:
        for feat in feature_names:
            train_path = f"dataset/synth_ts_data/train_{dataset}_{feat}.csv"
            test_path = f"dataset/synth_ts_data/test_{dataset}_{feat}.csv"
            output_file = f"results/{dataset}_{feat}.txt"
            os.makedirs("results", exist_ok=True)
            gpu_id = gpu_ids[i % len(gpu_ids)] # Assign GPUs in a round-robin fashion
            experiment_args.append((train_path, test_path, output_file, gpu_id))
            i += 1
    
    # divide the number of experiments by the max number of parallel processes
    chunks = [experiment_args[i:i+nbr_workers] for i in range(0, len(experiment_args), nbr_workers)]

    for chunk in chunks:
        processes = []
        for args in chunk:
            process = mp.Process(target=main, args=args)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    print("Done!")
