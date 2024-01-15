import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from aizynthfinder.aizynthfinder import AiZynthFinder
import pickle
import json
from tqdm import tqdm
import argparse
import numpy as np

def predict(chunk_id):
    np.random.seed(0)
    filename = 'regen_routes/retrostar.yml'
    # filename = 'retrostar.yml'
    finder = AiZynthFinder(configfile=filename)
    finder.stock.select(f"n1")
    print(len(finder.stock))
    finder.expansion_policy.select(f"uspto")
    # test_molecules = open(f'dataset/n1-targets.txt').read().split('\n')
    test_molecules = open(f'../data/n1-targets.txt').read().split('\n')
    chunk_size = 1000
    chunk_start = chunk_id * chunk_size
    chunk_end = (chunk_id + 1) * chunk_size
    if chunk_end > len(test_molecules):
        chunk_end = len(test_molecules)
   
    test_idxes = np.random.choice([i for i in range(10000)], 1000)
    for idx, test_idx in enumerate(tqdm(test_idxes)):
        # if chunk_start+idx in existed_number:
        #    continue
        finder.target_smiles = test_molecules[test_idx]
        finder.tree_search()
        finder.build_routes()
        stat = finder.extract_statistics()
        routes = [route['reaction_tree'].to_dict() for route in finder.routes]
        with open(f'../routes/n1_retrostar/routes_{test_idx}.json', 'w') as f:
            json.dump(routes, f)
        with open(f'../routes/n1_retrostar/statistics_{test_idx}.json', 'w') as f:
            json.dump(stat, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--data_id', type=int, default=0)
    args = parser.parse_args()  
    predict(args.chunk_id)
        