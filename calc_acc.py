import json
import pickle
import numpy as np
from tqdm import trange, tqdm
from BLEU_utils import *


all_routes = pickle.load(open('data/all_routes.pickle', 'rb'))
n5_routes = json.load(open('data/n5-routes.json'))
n1_routes = json.load(open('data/n1-routes.json'))
all_templates = json.load(open('../teamdrive/projects/n5routes/templates/all_routes_templates_1_0_0.json'))
golden_dict = pickle.load(open('../teamdrive/projects/n5routes/templates/golden_dict.pickle', 'rb'))
n5_bigrams = pickle.load(open('data/vocab_bigrams_n5.pkl', 'rb'))
n1_bigrams = pickle.load(open('data/vocab_bigrams_n1.pkl', 'rb'))


def calc_acc(route_type, route_name, add_golden=True, neg_bigram=set(), searchable=True):
    # neg_bigram = json.load(open('analyze_10_fold/filtered_negative_bigrams_s1_v8.json', 'r'))
    # neg_bigram = set([tuple(bigram) for bigram in neg_bigram])
    print(route_name, route_type, add_golden, len(neg_bigram))
    original_1, original_5, original_10 = 0, 0, 0
    original_1_worst, original_5_worst, original_10_worst = 0, 0, 0
    new_1, new_5, new_10 = 0, 0, 0
    new_1_worst, new_5_worst, new_10_worst = 0, 0, 0
    prob_1, prob_5, prob_10 = 0, 0, 0
    if route_type == 'n5routes':
        golden_routes = n5_routes
        pos_bigram = n5_bigrams
    else:
        golden_routes = n1_routes
        pos_bigram = n1_bigrams
    for i in range(10000):
        try:
            routes = json.load(open(f'../routes/{route_type}/{route_name}/routes_{i}.json'))
        except Exception as e:
            # print(e)
            # print(i)
            continue
        golden_route = golden_routes[i]
        golden_set = extract_mol_set(golden_route)
        candidate_sets = [extract_mol_set(r) for r in routes]
        golden_exist = golden_set in candidate_sets
        if add_golden:
            golden_score = route_score(golden_route)
            candidate_scores = [route_score(r) for r in routes]
            candidate_bleu = route_score_bleu([golden_route] + routes, neg_bigram, pos_bigram, all_templates)
            golden_bleu = candidate_bleu[0]
            candidate_bleu = candidate_bleu[1:]
            golden_prob = route_score_probability_golden(golden_route, golden_dict, searchable=searchable)
            if golden_prob == 0:
                continue
            candidate_probs = [route_score_probability(r) for r in routes]
        else:
            if golden_exist:
                golden_idx = candidate_sets.index(golden_set)
            else:
                continue
            candidate_scores = [route_score(r) for r in routes]
            candidate_bleu = route_score_bleu(routes, neg_bigram, pos_bigram, all_templates)
            golden_score = candidate_scores[golden_idx]
            golden_bleu = candidate_bleu[golden_idx]
            candidate_probs = [route_score_probability(r) for r in routes]
            golden_prob = candidate_probs[golden_idx]
        if sum(np.array(candidate_scores) < golden_score) < 1:
            original_1 += 1
        if sum(np.array(candidate_scores) <= golden_score) < 1 + golden_exist:
            original_1_worst += 1
        if sum(np.array(candidate_scores) < golden_score) < 5:
            original_5 += 1
        if sum(np.array(candidate_scores) <= golden_score) < 5 + golden_exist:
            original_5_worst += 1
        if sum(np.array(candidate_scores) < golden_score) < 10:
            original_10 += 1
        if sum(np.array(candidate_scores) <= golden_score) < 10 + golden_exist:
            original_10_worst += 1
        if sum(np.array(candidate_bleu) < golden_bleu) < 1:
            new_1 += 1
        if sum(np.array(candidate_bleu) <= golden_bleu) < 1 + golden_exist:
            new_1_worst += 1
        if sum(np.array(candidate_bleu) < golden_bleu) < 5:
            new_5 += 1
        if sum(np.array(candidate_bleu) <= golden_bleu) < 5 + golden_exist:
            new_5_worst += 1
        if sum(np.array(candidate_bleu) < golden_bleu) < 10:
            new_10 += 1
        if sum(np.array(candidate_bleu) <= golden_bleu) < 10 + golden_exist:
            new_10_worst += 1
        if sum(np.array(candidate_probs) > golden_prob) < 1:
            prob_1 += 1
        if sum(np.array(candidate_probs) > golden_prob) < 5:
            prob_5 += 1
        if sum(np.array(candidate_probs) > golden_prob) < 10:
            prob_10 += 1
        
    print(original_1, original_5, original_10)
    print(original_1_worst, original_5_worst, original_10_worst)
    print('\n')
    print(new_1, new_5, new_10)
    print(new_1_worst, new_5_worst, new_10_worst)
    print('\n')
    print(prob_1, prob_5, prob_10)
    print('\n')
    print('-'*20)

calc_acc('n5routes', 'mcts_v2')
calc_acc('n1routes', 'mcts_v2')
calc_acc('n5routes', 'retrostar_v2')
calc_acc('n1routes', 'retrostar_v2')