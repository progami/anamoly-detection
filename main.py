import os
import argparse
from utils.utils import fix_seed
from utils.utils import translate_sequence
from config.search_space import *
from config.controller_params import *
from config.autoencoder_params import *
from models.controller import PolicyGradientAgent
from models.controller import ModelsHistory


parser = argparse.ArgumentParser(description='Execute RLNAS for given dataset.')
parser.add_argument('--train_dataset_path', type=str, default='dataset/synth_ts_data/train_IWESTM40_Humidity_%.csv', help='Path to the training dataset')
parser.add_argument('--test_dataset_path', type=str, default='dataset/synth_ts_data/test_IWESTM40_Humidity_%.csv', help='Path to the testing dataset')
parser.add_argument('--output_file', type=str, default='results.txt', help='Path to the output file to save the results to')
args = parser.parse_args()


def main(train_dataset_path, test_dataset_path, output_file, gpu_id=0):
    fix_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    agent = PolicyGradientAgent(learning_rate, gamma, train_dataset_path, test_dataset_path)
    history = ModelsHistory(n_models=history_size)
    history.init_history()
    fitness_hist = []

    # training loop
    for ep in range(n_ep):
        print(f"===== episode: {ep+1}/{n_ep} =====")
        hist = history.build_history()
        full_arch, tokens_logprobs = agent.get_action(hist)
        part_arch = [0 for _ in range(number_blocks*number_features_per_block)]
        rewards = []
        for i, token in enumerate(full_arch):
            print(f"----- ep: {ep+1}/{n_ep} -> action: {i+1}/{len(full_arch)} -----")
            part_arch[i] = token
            fitness = agent.eval_arch(part_arch)
            rewards.append(fitness)
            print(f"part_arch: {translate_sequence(part_arch)}\n> fitness: {fitness}\n")
        print(f"full_arch: {translate_sequence(full_arch)}\n> fitness: {fitness}\n")  
        fitness_hist.append(fitness)
        history.add(fitness, full_arch) # fitness of full arch (all tokens)
        loss = agent.calc_fitness_loss(rewards, tokens_logprobs)
        agent.update_policy(loss)

    # evaluates best discovered architecture on test data
    best_arch = history.sample(1)[0]
    f1, precision, recall = agent.eval_arch(best_arch, "test", return_metrics=True)
    print(f"---\nbest arch: {translate_sequence(best_arch)}")
    print(f"fitness on test data: f1={f1}, precision={precision}, recall={recall}")

    # save into datasets results file
    print(f"fitness of best architecture on test data: {f1} -> {translate_sequence(best_arch)}\n---\n", file=open(output_file, "a"))
    folder = "weather_1b"
    os.makedirs(folder, exist_ok=True)
    print(f"{train_dataset_path.split('/')[-1]}\t{f1}\t{precision}\t{recall}", file=open(f"{folder}/results.txt", "a"))

    # save best architecture
    for fit in history.buffer.keys():
        arch = history.buffer[fit]
        print(f"fitness={fit} -> arch={translate_sequence(arch)}\n", file=open(output_file, "a"))
    print(f"rewards evolution = {fitness_hist}\n", file=open(output_file, "a"))
    
    print(f"===== Done! =====")


if __name__=="__main__":
    main(args.train_dataset_path, args.test_dataset_path, args.output_file)
    