import numpy as np
import json
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from BLEU_utils import *
import argparse
from transformers import logging
logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year_split', type=int, default=2016)
    args = parser.parse_args()
    template_radius = args.template_radius
    year_split = args.year_split
    freq_threshold = args.freq_threshold
    pmi_threshold = args.pmi_threshold
    rxnmapper = RXNMapper()

    print('\nExperiment on PaRoutes dataset')
    print('-' * 10)
    print('Template radius:', template_radius)
    print('Vocab on all routes')
    paroute_routes = pickle.load(open('data/all_routes.pickle', 'rb'))
    paroute_templates = json.load(open('../teamdrive/projects/n5routes/templates/all_routes_templates_1_0_0.json'))
    all_ngram_rxns, all_ngram_templates, all_rxns = build_vocab_paroutes(paroute_routes, paroute_templates)
    all_temp_dict = Counter([paroute_templates[rxn] for rxn in all_rxns])
    set_ngram_rxns = [set(ngram_rxns) for ngram_rxns in all_ngram_rxns]
    set_ngram_templates = [set(ngram_templates) for ngram_templates in all_ngram_templates]
    print('Number of n-gram rxns for n in [2, 3, 4, 5]:', [len(ngram_rxns) for ngram_rxns in set_ngram_rxns])
    print('Number of n-gram templates for n in [2, 3, 4, 5]:', [len(ngram_templates) for ngram_templates in set_ngram_templates])
    print('-' * 10)
    
    print('Vocab on routes before', year_split)
    all_ngram_rxns, all_ngram_templates, all_rxns = build_vocab_paroutes(paroute_routes, paroute_templates, year_split)
    all_temp_dict = Counter([paroute_templates[rxn] for rxn in all_rxns])
    set_ngram_rxns = [set(ngram_rxns) for ngram_rxns in all_ngram_rxns]
    set_ngram_templates = [set(ngram_templates) for ngram_templates in all_ngram_templates]
    bigram_counter = Counter(all_ngram_templates[0])
    bigram_pmi = {}
    for k, v in bigram_counter.items():
        if v < freq_threshold:
            continue
        bigram_pmi[k] = (v, np.log((v / len(bigram_counter)) / (all_temp_dict[k[0]] * all_temp_dict[k[1]] / len(all_ngram_templates[0]) ** 2)))
    min_pmi, max_pmi = min([v[1] for v in bigram_pmi.values()]), max([v[1] for v in bigram_pmi.values()])
    for k, v in bigram_pmi.items():
        bigram_pmi[k] = (v[0], (v[1] - min_pmi) / (max_pmi - min_pmi))
    bigram_pmi = {k: v for k, v in bigram_pmi.items() if v[1] > pmi_threshold}
    print(f'Number of n-gram rxns for n in [2, 3, 4, 5] in patents before {year_split}:', [len(ngram_rxns) for ngram_rxns in set_ngram_rxns])
    print(f'Number of n-gram templates for n in [2, 3, 4, 5] in patents before {year_split}:', [len(ngram_templates) for ngram_templates in set_ngram_templates])
    
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    bleu_pmi_score = []
    for route in paroute_routes:
        rxn_nodes = extract_rxn_paroutes(route)
        if int(rxn_nodes[0]['metadata']['ID'].split(';')[-1]) < year_split:
            continue
        bigram, trigram, tetragram, pentagram = extract_bigram(rxn_nodes), extract_trigram(rxn_nodes), extract_tetragram(rxn_nodes), extract_pentagram(rxn_nodes)
        if len(bigram) > 0:
            bleu_score[0].append(sum([bi in set_ngram_rxns[0] for bi in bigram]) / len(bigram))
            bigram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in bigram]
            bleu_template_score[0].append(sum([bi in set_ngram_templates[0] for bi in bigram_template]) / len(bigram_template))
            bleu_pmi_score.append(sum([bi in bigram_pmi for bi in bigram_template]) / len(bigram_template))
        if len(trigram) > 0:
            bleu_score[1].append(sum([tri in set_ngram_rxns[1] for tri in trigram]) / len(trigram))
            trigram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in trigram]
            bleu_template_score[1].append(sum([tri in set_ngram_templates[1] for tri in trigram_template]) / len(trigram_template))
        if len(tetragram) > 0:
            bleu_score[2].append(sum([tetra in set_ngram_rxns[2] for tetra in tetragram]) / len(tetragram))
            tetragram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in tetragram]
            bleu_template_score[2].append(sum([tetra in set_ngram_templates[2] for tetra in tetragram_template]) / len(tetragram_template))
        if len(pentagram) > 0:
            bleu_score[3].append(sum([penta in set_ngram_rxns[3] for penta in pentagram]) / len(pentagram))
            pentagram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in pentagram]
            bleu_template_score[3].append(sum([penta in set_ngram_templates[3] for penta in pentagram_template]) / len(pentagram_template))
    
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in patents after {year_split}:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'Bigram score with PMI > {pmi_threshold} and frequency > {freq_threshold} in patents after {year_split}:', np.mean(bleu_pmi_score))
    print('-' * 10)

    print('Vocab on random 90 percent train/10 percent test split')
    train_routes, test_routes = train_test_split(paroute_routes, test_size=0.1, random_state=42)
    all_ngram_rxns, all_ngram_templates, all_rxns = build_vocab_paroutes(train_routes, paroute_templates)
    all_temp_dict = Counter([paroute_templates[rxn] for rxn in all_rxns])
    set_ngram_rxns = [set(ngram_rxns) for ngram_rxns in all_ngram_rxns]
    set_ngram_templates = [set(ngram_templates) for ngram_templates in all_ngram_templates]
    bigram_counter = Counter(all_ngram_templates[0])
    bigram_pmi = {}
    for k, v in bigram_counter.items():
        if v < freq_threshold:
            continue
        bigram_pmi[k] = (v, np.log((v / len(bigram_counter)) / (all_temp_dict[k[0]] * all_temp_dict[k[1]] / len(all_ngram_templates[0]) ** 2)))
    min_pmi, max_pmi = min([v[1] for v in bigram_pmi.values()]), max([v[1] for v in bigram_pmi.values()])
    for k, v in bigram_pmi.items():
        bigram_pmi[k] = (v[0], (v[1] - min_pmi) / (max_pmi - min_pmi))
    bigram_pmi = {k: v for k, v in bigram_pmi.items() if v[1] > pmi_threshold}
    print('Number of n-gram rxns for n in [2, 3, 4, 5] in random train set:', [len(ngram_rxns) for ngram_rxns in set_ngram_rxns])
    print('Number of n-gram templates for n in [2, 3, 4, 5] in random train set:', [len(ngram_templates) for ngram_templates in set_ngram_templates])
    
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    for route in test_routes:
        rxn_nodes = extract_rxn_paroutes(route)
        bigram, trigram, tetragram, pentagram = extract_bigram(rxn_nodes), extract_trigram(rxn_nodes), extract_tetragram(rxn_nodes), extract_pentagram(rxn_nodes)
        if len(bigram) > 0:
            bleu_score[0].append(sum([bi in set_ngram_rxns[0] for bi in bigram]) / len(bigram))
            bigram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in bigram]
            bleu_template_score[0].append(sum([bi in set_ngram_templates[0] for bi in bigram_template]) / len(bigram_template))
            bleu_pmi_score.append(sum([bi in bigram_pmi for bi in bigram_template]) / len(bigram_template))
        if len(trigram) > 0:
            bleu_score[1].append(sum([tri in set_ngram_rxns[1] for tri in trigram]) / len(trigram))
            trigram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in trigram]
            bleu_template_score[1].append(sum([tri in set_ngram_templates[1] for tri in trigram_template]) / len(trigram_template))
        if len(tetragram) > 0:
            bleu_score[2].append(sum([tetra in set_ngram_rxns[2] for tetra in tetragram]) / len(tetragram))
            tetragram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in tetragram]
            bleu_template_score[2].append(sum([tetra in set_ngram_templates[2] for tetra in tetragram_template]) / len(tetragram_template))
        if len(pentagram) > 0:
            bleu_score[3].append(sum([penta in set_ngram_rxns[3] for penta in pentagram]) / len(pentagram))
            pentagram_template = [tuple([paroute_templates[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in pentagram]
            bleu_template_score[3].append(sum([penta in set_ngram_templates[3] for penta in pentagram_template]) / len(pentagram_template))
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in random test set:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'n-gram reaction-BLEU score for n in [2, 3, 4, 5] in random test set:', [np.mean(bleu_score[i]) for i in range(4)])
    print('-' * 10)


    paroute_routes = pickle.load(open('data/all_routes.pickle', 'rb'))
    paroute_templates = json.load(open('../teamdrive/projects/n5routes/templates/all_routes_templates_1_0_0.json'))
    all_ngram_rxns, all_ngram_templates, all_rxns = build_vocab_paroutes(paroute_routes, paroute_templates)
    all_temp_dict = Counter([paroute_templates[rxn] for rxn in all_rxns])
    set_ngram_rxns = [set(ngram_rxns) for ngram_rxns in all_ngram_rxns]
    set_ngram_templates = [set(ngram_templates) for ngram_templates in all_ngram_templates]
    
    print('Test on Golden routes in 190 Retro* target molecules')
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    bleu_pmi_score = []
    golden_routes = pickle.load(open('test_routes/routes_retrostar_golden.pkl', 'rb'))
    for route in tqdm(golden_routes):
        all_ngram_rxns, all_ngram_templates = get_test_ngram(route, rxnmapper, template_radius)
        for i in range(4):
            if len(all_ngram_rxns[i]) > 0:
                if i == 0:
                    bleu_pmi_score.append(sum([bi in bigram_pmi for bi in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
                bleu_score[i].append(sum([n_gram_rxn in set_ngram_rxns[i] for n_gram_rxn in all_ngram_rxns[i]]) / len(all_ngram_rxns[i]))
                bleu_template_score[i].append(sum([n_gram_rxn in set_ngram_templates[i] for n_gram_rxn in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in golden routes:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'Bigram PMI score in golden routes:', np.mean(bleu_pmi_score))
    print('-' * 10)

    print('Test on Retro* predicted routes in 190 Retro* target molecules')
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    bleu_pmi_score = []
    retro_routes = json.load(open('test_routes/routes_retrostar.json'))['routes']
    retro_routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in retro_routes if isinstance(route, str)]
    for route in tqdm(retro_routes):
        all_ngram_rxns, all_ngram_templates = get_test_ngram(route, rxnmapper, template_radius)
        for i in range(4):
            if len(all_ngram_rxns[i]) > 0:
                if i == 0:
                    bleu_pmi_score.append(sum([bi in bigram_pmi for bi in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
                bleu_score[i].append(sum([n_gram_rxn in set_ngram_rxns[i] for n_gram_rxn in all_ngram_rxns[i]]) / len(all_ngram_rxns[i]))
                bleu_template_score[i].append(sum([n_gram_rxn in set_ngram_templates[i] for n_gram_rxn in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in Retro* routes:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'Bigram PMI score in Retro* routes:', np.mean(bleu_pmi_score))
    print('-' * 10)

    print('Test on Retro*+ predicted routes in 190 Retro* target molecules')
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    bleu_pmi_score = []
    retro_routes = json.load(open('test_routes/routes_retrostarplus.json'))['routes']
    retro_routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in retro_routes if isinstance(route, str)]
    for route in tqdm(retro_routes):
        all_ngram_rxns, all_ngram_templates = get_test_ngram(route, rxnmapper, template_radius)
        for i in range(4):
            if len(all_ngram_rxns[i]) > 0:
                if i == 0:
                    bleu_pmi_score.append(sum([bi in bigram_pmi for bi in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
                bleu_score[i].append(sum([n_gram_rxn in set_ngram_rxns[i] for n_gram_rxn in all_ngram_rxns[i]]) / len(all_ngram_rxns[i]))
                bleu_template_score[i].append(sum([n_gram_rxn in set_ngram_templates[i] for n_gram_rxn in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in Retro*+ routes:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'Bigram PMI score in Retro*+ routes:', np.mean(bleu_pmi_score))
    print('-' * 10)

    print('Test on EG-MCTS predicted routes in 190 Retro* target molecules')
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    bleu_pmi_score = []
    retro_routes = json.load(open('test_routes/routes_eg_mcts.json'))['routes']
    retro_routes = [[rxn.split('>')[0] + '>>' + rxn.split('>')[-1] for rxn in route.split('|')] for route in retro_routes if isinstance(route, str)]
    for route in tqdm(retro_routes):
        all_ngram_rxns, all_ngram_templates = get_test_ngram(route, rxnmapper, template_radius)
        for i in range(4):
            if len(all_ngram_rxns[i]) > 0:
                if i == 0:
                    bleu_pmi_score.append(sum([bi in bigram_pmi for bi in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
                bleu_score[i].append(sum([n_gram_rxn in set_ngram_rxns[i] for n_gram_rxn in all_ngram_rxns[i]]) / len(all_ngram_rxns[i]))
                bleu_template_score[i].append(sum([n_gram_rxn in set_ngram_templates[i] for n_gram_rxn in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in EG-MCTS routes:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'Bigram PMI score in EG-MCTS routes:', np.mean(bleu_pmi_score))
    print('-' * 10)

    print('Test on RetroGraph predicted routes in 190 Retro* target molecules')
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    bleu_pmi_score = []
    golden_routes = json.load(open('test_routes/routes_retrograph.json'))
    for route in tqdm(golden_routes):
        all_ngram_rxns, all_ngram_templates = get_test_ngram(route, rxnmapper, template_radius)
        for i in range(4):
            if len(all_ngram_rxns[i]) > 0:
                if i == 0:
                    bleu_pmi_score.append(sum([bi in bigram_pmi for bi in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
                bleu_score[i].append(sum([n_gram_rxn in set_ngram_rxns[i] for n_gram_rxn in all_ngram_rxns[i]]) / len(all_ngram_rxns[i]))
                bleu_template_score[i].append(sum([n_gram_rxn in set_ngram_templates[i] for n_gram_rxn in all_ngram_templates[i]]) / len(all_ngram_templates[i]))
    print(f'n-gram template-BLEU score for n in [2, 3, 4, 5] in RetroGraph routes:', [np.mean(bleu_template_score[i]) for i in range(4)])
    print(f'Bigram PMI score in RetroGraph routes:', np.mean(bleu_pmi_score))
    print('-' * 10)