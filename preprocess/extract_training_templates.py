import json
import pickle
from tqdm import tqdm
from rxnutils.chem.reaction import ChemicalReaction
from multiprocessing import Pool
from rxnmapper import RXNMapper
from transformers import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chunk_id', type=int, default=0)
args = parser.parse_args()

rxnmapper = RXNMapper()

chunk_num = 20
retro_train_routes = pickle.load(open('../data/routes_train.pkl', 'rb'))
chunk_start = len(retro_train_routes) // chunk_num * args.chunk_id
chunk_end = min(len(retro_train_routes) // chunk_num * (args.chunk_id + 1), len(retro_train_routes))
train_rxns = [i for j in retro_train_routes[chunk_start:chunk_end] for i in j]
train_rxns = list(set(train_rxns))
train_temp_dict_0 = {}
train_temp_dict_1 = {}
train_temp_dict_2 = {}
for idx, rxn in enumerate(train_rxns):
    if idx % 1000 == 0:
        print(idx, args.chunk_id)
    try:
        forward_rxn = rxn.split('>>')[-1] + '>>' + rxn.split('>>')[0]
        mapped_rxn = rxnmapper.get_attention_guided_atom_maps([forward_rxn])[0]['mapped_rxn']
        template_0 = ChemicalReaction(mapped_rxn).generate_reaction_template(radius=0, expand_ring=False, expand_hetero=False)[1].smarts
        template_1 = ChemicalReaction(mapped_rxn).generate_reaction_template(radius=1, expand_ring=False, expand_hetero=False)[1].smarts
        template_2 = ChemicalReaction(mapped_rxn).generate_reaction_template(radius=2, expand_ring=False, expand_hetero=False)[1].smarts
    except Exception as e:
        print(e)
        template_0 = ''
        template_1 = ''
        template_2 = ''
    train_temp_dict_0[rxn] = template_0
    train_temp_dict_1[rxn] = template_1
    train_temp_dict_2[rxn] = template_2
json.dump(train_temp_dict_0, open(f'../data/{args.chunk_id}_train_routes_templates_0_0_0.json', 'w'))
json.dump(train_temp_dict_1, open(f'../data/{args.chunk_id}_train_routes_templates_1_0_0.json', 'w'))
json.dump(train_temp_dict_2, open(f'../data/{args.chunk_id}_train_routes_templates_2_0_0.json', 'w'))