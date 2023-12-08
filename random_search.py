import random
from tqdm import tqdm
from utils.utils import fix_seed
from config.search_space import *
from config.controller_params import *
from config.autoencoder_params import *
from models.autoencoder import Architecture


def eval_dataset(dataset, feat, max_iter=200):
    train_dataset_path = f"dataset/synth_ts_data/train_{dataset}_{feat}.csv"
    test_dataset_path = f"dataset/synth_ts_data/test_{dataset}_{feat}.csv"

    def get_random_arch():
        number_blocks = 2
        model_arch = []
        for _ in range(number_blocks):
            model_arch.append(random.randint(0, len(conv_n_out_channels)-1))
            model_arch.append(random.randint(0, len(conv_kernel_size)-1))
            model_arch.append(random.randint(0, len(pool_type)-1))
            model_arch.append(random.randint(0, len(pool_kernel_size)-1))
            model_arch.append(random.randint(0, len(norm_type)-1))
            model_arch.append(random.randint(0, len(act_funcs)-1))
        return model_arch

    tried_archs = []
    for iter in tqdm(range(1, 1+max_iter)):
        model_arch = get_random_arch()
        model = Architecture(model_arch, train_dataset_path, test_dataset_path)
        model.train()
        fitness = model.fitness(data_type="val")
        tried_archs.append((fitness, model_arch))
        del model

        if iter in [5, 10, 50, 100, 200]:
            tried_archs.sort(key=lambda x: x[0], reverse=True)
            best_arch = tried_archs[0][1]
            final_model = Architecture(best_arch, train_dataset_path, test_dataset_path)
            final_model.train()
            f1, precision, recall = final_model.fitness(data_type="test", return_metrics=True)
            print(f"dataset={dataset}\t feat={feat}\t nbr_iter={iter}\t f1={f1}\t precision={precision}\t recall={recall}", file=open("RS_results.txt", "a"))


fix_seed()

for dataset in ["IHAMPS1", "IMONTR17", "IMONTR172", "IQUEBECM21", "IWESTM40"]:
    for feat in ["Temperature_C", "Humidity_%", "Pressure_hPa"]:
        eval_dataset(dataset, feat)

for dataset in ["GAS_CONCENTRATION", "HUMIDITY", "ILLUMINANCE", "POWER", "TEMPERATURE"]:
    eval_dataset(dataset, "value")
