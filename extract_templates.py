import json
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rxnutils.chem.reaction import ChemicalReaction
from BLEU_utils import *
from multiprocessing import Pool
from rxnmapper import RXNMapper
import os
from transformers import logging
import argparse

logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_radius', type=int, default=1)
    args = parser.parse_args()

    all_rxns = []
    routes = json.load(open('test_routes/routes_retrostarplus.json'))['routes']
    routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in routes if isinstance(route, str)]
    all_rxns.extend([rxn for route in routes for rxn in route])
    routes = json.load(open('test_routes/routes_retrostar.json'))['routes']
    routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in routes if isinstance(route, str)]
    all_rxns.extend([rxn for route in routes for rxn in route])
    golden_routes = pickle.load(open('test_routes/routes_golden.pkl', 'rb'))
    all_rxns.extend([rxn for route in golden_routes for rxn in route])
    routes = json.load(open('test_routes/routes_egmcts.json'))['routes']
    routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in routes if isinstance(route, str)]
    all_rxns.extend([rxn for route in routes for rxn in route])
    golden_routes = json.load(open('test_routes/routes_retrograph.json'))
    all_rxns.extend([rxn for route in golden_routes for rxn in route])

    all_rxns = list(set(all_rxns))
    print(len(all_rxns))

    all_temp_dict = {}

    '''routes = json.load(open('data/150k_routes.json'))
    all_ngram_rxns = build_vocab_paroutes(routes, None)
    paroutes_rxns = all_ngram_rxns[0]
    paroutes_rxns = list(set([i for j in paroutes_rxns for i in j]))'''

    # There is no need to re-mapping the reactions in PaRoutes, while you can use the following code to map the reactions.
    '''if os.path.exists('data/paroutes_mapped_rxn_dict.json'):
        mapped_rxn_dict = json.load(open('data/paroutes_mapped_rxn_dict.json'))
        all_ngram_rxns, all_ngram_templates = build_vocab_paroutes(routes, mapped_rxn_dict)
    else:
        all_ngram_rxns = build_vocab_paroutes(routes, None)
        paroutes_rxns = all_ngram_rxns[0]
        paroutes_rxns = list(set([i for j in paroutes_rxns for i in j]))
        mapped_rxn_dict = {}
        def mapping_rxn(idx):
            rxnmapper = RXNMapper()
            start = idx * 100
            end = (idx + 1) * 100 if (idx + 1) * 100 < len(paroutes_rxns) else len(paroutes_rxns)
            rxns = paroutes_rxns[start:end]
            current_dict = {}
            for rxn in rxns:
                try:
                    current_dict[rxn] = rxnmapper.get_attention_guided_atom_maps([rxn])['mapped_rxn']
                except:
                    current_dict[rxn] = ''
            return current_dict
        with Pool(16) as p:
            for res in tqdm(p.imap(mapping_rxn, (i for i in range(len(paroutes_rxns)//100))), total=len(paroutes_rxns)//100, mininterval=10):
                mapped_rxn_dict.update(res)
        json.dump(mapped_rxn_dict, open('data/paroutes_mapped_rxn_dict.json', 'w'))
        all_ngram_rxns, all_ngram_templates = build_vocab_paroutes(routes, mapped_rxn_dict)'''

    paroutes_temp_dict = {}
    rxnmapper = RXNMapper()
    def work(rxn):
        try:
            forward_rxn = rxn.split('>>')[-1] + '>>' + rxn.split('>>')[0]
            mapped_rxn = rxnmapper.get_attention_guided_atom_maps([forward_rxn])
            print(mapped_rxn)
            template = ChemicalReaction(mapped_rxn).generate_reaction_template(radius=args.template_radius, expand_ring=False, expand_hetero=False)[1].smarts
        except Exception as e:
            print(e)
            template = ''
        return {rxn: template}



    '''with Pool(24) as p:
        for res in tqdm(p.imap(work, (rxn for rxn in all_rxns)), total=len(all_rxns), mininterval=10):
            paroutes_temp_dict.update(res)'''
    for rxn in tqdm(all_rxns):
        try:
            forward_rxn = rxn.split('>>')[-1] + '>>' + rxn.split('>>')[0]
            mapped_rxn = rxnmapper.get_attention_guided_atom_maps([forward_rxn])[0]['mapped_rxn']
            template = ChemicalReaction(mapped_rxn).generate_reaction_template(radius=args.template_radius, expand_ring=False, expand_hetero=False)[1].smarts
        except Exception as e:
            print(e)
            template = ''
        paroutes_temp_dict[rxn] = template
    json.dump(paroutes_temp_dict, open(f'template/test_routes_templates_{args.template_radius}_0_0.json', 'w'))


    # Pistachio
    '''
    routes = json.load(open('data/pistachio_routes.json'))
    all_ngram_rxns, all_ngram_templates = build_vocab_pistachio(routes)
    pistachio_rxns = all_ngram_rxns[0]
    pistachio_rxns = list(set([i for j in pistachio_rxns for i in j]))
    piastachio_map = json.load(open('data/pistachio_mapped_rxn_dict.json'))
    pistachio_rxns = [piastachio_map[i] for i in pistachio_rxns]
    pistachio_temp_dict = []

    def work(rxn):
        try:
            template = ChemicalReaction(rxn).generate_reaction_template(radius=1, expand_ring=False, expand_hetero=False)[1].smarts
        except:
            template = ''
        return template



    with Pool(48) as p:
        for res in tqdm(p.imap(work, (rxn for rxn in pistachio_rxns)), total=len(pistachio_rxns), mininterval=10):
            pistachio_temp_dict.append(res)
    json.dump(pistachio_temp_dict, open('template/pistachio_templates_1_0_0.json', 'w'))

    def work(rxn):
        try:
            template = ChemicalReaction(rxn).generate_reaction_template(radius=0, expand_ring=False, expand_hetero=False)[1].smarts
        except:
            template = ''
        return template



    with Pool(48) as p:
        for res in tqdm(p.imap(work, (rxn for rxn in pistachio_rxns)), total=len(pistachio_rxns), mininterval=10):
            pistachio_temp_dict.append(res)
    json.dump(pistachio_temp_dict, open('template/pistachio_templates_0_0_0.json', 'w'))'''
