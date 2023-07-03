import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from aizynthfinder.aizynthfinder import AiZynthFinder
import pickle
import json
from tqdm import tqdm
import argparse
import multiprocessing

def predict(chunk_id):
    filename = 'mcts.yml'
    # filename = 'retrostar.yml'
    finder = AiZynthFinder(configfile=filename)
    finder.stock.select(f"n5")
    finder.expansion_policy.select(f"uspto")
    # test_molecules = open(f'dataset/n1-targets.txt').read().split('\n')
    test_molecules = open(f'dataset/n5-targets.txt').read().split('\n')
    chunk_size = 7
    chunk_start = chunk_id * chunk_size
    chunk_end = (chunk_id + 1) * chunk_size
    if chunk_end > len(test_molecules):
        chunk_end = len(test_molecules)
    test_molecules = test_molecules[chunk_start:chunk_end]
    for idx, test_mol in enumerate(tqdm(test_molecules)):
        if chunk_start+idx in existed_number:
           continue
        finder.target_smiles = test_mol
        finder.tree_search()
        finder.build_routes()
        stat = finder.extract_statistics()
        routes = [route['reaction_tree'].to_dict() for route in finder.routes]
        with open(f'/teamdrive/projects/n5routes/exclude_neg_5_5/routes_{chunk_start+idx}.json', 'w') as f:
            json.dump(routes, f)
        with open(f'/teamdrive/projects/n5routes/exclude_neg_5_5/statistics_{chunk_start+idx}.json', 'w') as f:
            json.dump(stat, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--data_id', type=int, default=0)
    args = parser.parse_args()
    existed = os.listdir(f'/teamdrive/projects/n5routes/exclude_neg_5_5/')
    existed_number = set([int(name.split('_')[1].split('.')[0]) for name in existed if name.split('_')[0] == 'routes'])
    print(len(existed_number))
    
    predict(args.chunk_id)
        