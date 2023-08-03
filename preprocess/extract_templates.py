import json
import pickle
from tqdm import tqdm
from rxnutils.chem.reaction import ChemicalReaction
from multiprocessing import Pool
from rxnmapper import RXNMapper
from transformers import logging
import argparse

logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_radius', type=int, default=1)
    args = parser.parse_args()
    
    
    def extract_rxns(route):
        rxn_nodes = []
        def preorder(root):
            if root.get('children') == None:
                return
            for child in root['children']:
                if child.get('type') == 'reaction':
                    rxn_nodes.append(child)
                preorder(child)
        preorder(route)
        return rxn_nodes
    
    rxnmapper = RXNMapper()

    test_rxns = []
    routes = json.load(open('../test_routes/routes_retrostarplus.json'))['routes']
    routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in routes if isinstance(route, str)]
    test_rxns.extend([rxn for route in routes for rxn in route])
    routes = json.load(open('../test_routes/routes_retrostar.json'))['routes']
    routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in routes if isinstance(route, str)]
    test_rxns.extend([rxn for route in routes for rxn in route])
    golden_routes = pickle.load(open('../test_routes/routes_golden.pkl', 'rb'))
    test_rxns.extend([rxn for route in golden_routes for rxn in route])
    routes = json.load(open('../test_routes/routes_egmcts.json'))['routes']
    routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in routes if isinstance(route, str)]
    test_rxns.extend([rxn for route in routes for rxn in route])
    golden_routes = json.load(open('../test_routes/routes_retrograph.json'))
    test_rxns.extend([rxn for route in golden_routes for rxn in route])

    test_rxns = list(set(test_rxns))
    print(len(test_rxns))

    test_temp_dict_0 = {}
    test_temp_dict_1 = {}
    test_temp_dict_2 = {}
    for rxn in tqdm(test_rxns):
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
        test_temp_dict_0[rxn] = template_0
        test_temp_dict_1[rxn] = template_1
        test_temp_dict_2[rxn] = template_2
    
    json.dump(test_temp_dict_0, open(f'../data/test_routes_templates_0_0_0.json', 'w'))
    json.dump(test_temp_dict_1, open(f'../data/test_routes_templates_1_0_0.json', 'w'))
    json.dump(test_temp_dict_2, open(f'../data/test_routes_templates_2_0_0.json', 'w'))    



    all_routes = pickle.load(open('../data/all_routes.pickle', 'rb'))
    all_rxns = []
    for idx, route in enumerate(all_routes):
        rxn_nodes = extract_rxns(route)
        for rxn_nodes in rxn_nodes:
            all_rxns.append(rxn_nodes['metadata']['smiles'])
            
    n5_routes = json.load(open('../data/n5-routes.json'))
    n1_routes = json.load(open('../data/n1-routes.json'))
    for idx, route in enumerate(n5_routes + n1_routes):
        rxn_nodes = extract_rxns(route)
        for rxn_nodes in rxn_nodes:
            all_rxns.append(rxn_nodes['metadata']['smiles'])

    all_rxns = list(set(all_rxns))



    all_temp_dict = []

    def work(rxn):
        try:
            template = ChemicalReaction(rxn).generate_reaction_template(radius=args.template_radius, expand_ring=False, expand_hetero=False)[1].smarts
        except:
            template = ''
        return template
    with Pool(48) as p:
        for res in tqdm(p.imap(work, (rxn for rxn in all_rxns)), total=len(all_rxns), mininterval=10):
            all_temp_dict.append(res)

    all_rxn_dict = {}
    for (rxn, retro_temp) in zip(all_rxns, all_temp_dict):
        all_rxn_dict[rxn] = retro_temp
    json.dump(all_rxn_dict, open(f'../data/all_routes_templates_{args.template_radius}_0_0.json', 'w'))