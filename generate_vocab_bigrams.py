import json
import pickle
import numpy as np
from queue import Queue
from tqdm import tqdm
from rdkit import Chem
from collections import Counter
from tqdm import trange, tqdm
from BLEU_utils import *

all_routes = pickle.load(open('data/all_routes.pickle', 'rb'))
all_templates = json.load(open('data/all_routes_templates_1_0_0.json'))
n5_routes = json.load(open('data/n5-routes.json'))
n1_routes = json.load(open('data/n1-routes.json'))

n5_patents = set([extract_rxns(route)[0]['metadata']['ID'].split(';')[0] for route in n5_routes])
n1_patents = set([extract_rxns(route)[0]['metadata']['ID'].split(';')[0] for route in n1_routes])

def build_vocab(routes, template_dict, excluded_patents=None):
    all_ngram_rxns = [[] for _ in range(4)]
    all_rxns = []
    for idx, route in enumerate(routes):
        rxn_nodes = extract_rxns(route)
        if rxn_nodes[0]['metadata']['ID'].split(';')[0] in excluded_patents:
            continue
        for rxn_node in rxn_nodes:
            all_rxns.append(rxn_node['metadata']['smiles'])
        all_ngram_rxns[0].extend(extract_bigram(rxn_nodes))
        all_ngram_rxns[1].extend(extract_trigram(rxn_nodes))
        all_ngram_rxns[2].extend(extract_tetragram(rxn_nodes))
        all_ngram_rxns[3].extend(extract_pentagram(rxn_nodes))
    if template_dict is None:
        return all_ngram_rxns
    all_ngram_templates = [[tuple([template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in n_gram_rxns] for n_gram_rxns in all_ngram_rxns]
    return all_ngram_rxns, all_ngram_templates, all_rxns

n5_ngram_rxns, n5_ngram_templates, n5_rxns = build_vocab(n5_routes, all_templates, excluded_patents=n5_patents)
n5_bigrams = set(n5_ngram_templates[0])
n1_ngram_rxns, n1_ngram_templates, n1_rxns = build_vocab(n1_routes, all_templates, excluded_patents=n1_patents)
n1_bigrams = set(n1_ngram_templates[0])

pickle.dump(n5_bigrams, open('data/vocab_bigrams_n5.pkl', 'wb'))
pickle.dump(n1_bigrams, open('data/vocab_bigrams_n1.pkl', 'wb'))


