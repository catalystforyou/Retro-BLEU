import json
import pickle
from rdkit import Chem
from tqdm import trange, tqdm
import numpy as np
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction
import argparse


def get_template(mapped_rxn, radius=1):
    try:
        template = ChemicalReaction(mapped_rxn).generate_reaction_template(radius)[1].smarts
    except:
        template = mapped_rxn
    return template

def heuristic_score(length, ref_length):
    if length <= ref_length:
        return 1
    else:
        return ref_length / length

def get_single_retrobleu(route, pos_bigrams, neg_bigrams=None, template_radius=1, ref_length=3):
    rxnmapper = RXNMapper()
    test_bigrams = []
    if '.' in route[0].split('>>')[0]:
        backward_rxns = ['>>'.join([rxn.split('>')[-1], rxn.split('>')[0]]) for rxn in route]
        forward_rxns = route
    else:
        backward_rxns = route
        forward_rxns = ['>>'.join([rxn.split('>')[-1], rxn.split('>')[0]]) for rxn in route]
    precursors_dict = {rxn.split('>>')[0]:rxn for rxn in backward_rxns}
    mapped_rxns = rxnmapper.get_attention_guided_atom_maps(forward_rxns)
    mapped_rxns = [rxn['mapped_rxn'] for rxn in mapped_rxns]
    test_template_dict = {rxn:get_template(mapped_rxn, radius=template_radius) for (rxn, mapped_rxn) in zip(route, mapped_rxns)}
    for rxn in backward_rxns:
        reactants = rxn.split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                test_bigrams.append((test_template_dict[rxn], test_template_dict[precursors_dict[reactant]]))
    pos_ratio = sum([bi in pos_bigrams for bi in test_bigrams]) / len(test_bigrams)
    retro_bleu = np.exp(pos_ratio) + np.exp(heuristic_score(len(route), ref_length=ref_length))
    if neg_bigrams is not None:
        neg_ratio = sum([bi in neg_bigrams for bi in test_bigrams]) / len(test_bigrams)
        retro_bleu = np.exp(pos_ratio) + np.exp(heuristic_score(len(route), ref_length=ref_length)) - np.exp(neg_ratio)
    return retro_bleu

def get_batch_retrobleu(routes_path, pos_path=None, neg_path=None, template_radius=1, ref_length=3):
    routes = json.load(open(routes_path))
    if pos_path is None:
        pos_bigrams = pickle.load(open('data/vocab_bigrams_all.pkl', 'rb'))
        print(f'Using the {len(pos_bigrams)} default positive bigarms extracted from USPTO dataset.')
    else:
        pos_bigrams = pickle.load(open(pos_bigrams), 'rb')
        print(f'Loading {len(pos_bigrams)} positive bigarms extracted from f{pos_path}.')
    if neg_path is None:
        neg_bigrams = None
    else:
        neg_bigrams = pickle.load(open(neg_bigrams), 'rb')
    if ref_length < 3:
        print('Warning: Applying a reference length less than 3 may cause unexpected discrimination to longer routes.')
    batch_retrobleus = [get_single_retrobleu(route, pos_bigrams=pos_bigrams, neg_bigrams=neg_bigrams, template_radius=template_radius, ref_length=ref_length) for route in tqdm(routes)]
    return batch_retrobleus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for calculating Retro-BLEU')
    parser.add_argument('--routes', default='test_routes/routes_retrograph.json', help='The path of synthesis/retrosynthesis routes, should be loaded in json format.')
    parser.add_argument('--pos_path', default=None, help='The path of the positive bigrams.')
    parser.add_argument('--neg_path', default=None, help='The path of the negative bigrams.')
    parser.add_argument('--radius', type=int, default=1, help='The radius used when extracting reaction templates.')
    parser.add_argument('--ref_length', type=int, default=3, help='The reference length for evaluating the routes.')
    args = parser.parse_args()

    retro_bleus = get_batch_retrobleu(args.routes, args.pos_path, args.neg_path, args.radius, args.ref_length)
    print('The calculated retro-bleu score is', retro_bleus)