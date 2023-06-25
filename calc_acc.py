import json
import pickle
import numpy as np
from queue import Queue
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rxnutils.chem.reaction import ChemicalReaction
from collections import Counter
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from typing import Dict, Any, Set, List, Tuple

all_routes = pickle.load(open('data/all_routes.pickle', 'rb'))
n5_routes = json.load(open('data/n5-routes.json'))
n1_routes = json.load(open('data/n1-routes.json'))
# n5_3_routes = json.load(open('setup/n5-3-routes.json'))
n5_templates = json.load(open('../teamdrive/projects/n5routes/templates/n5_templates_1_0_0.json'))
# n5_templates_0 = json.load(open('../teamdrive/projects/n5routes/templates/n5_templates_0_0_0.json'))
n1_templates = json.load(open('../teamdrive/projects/n5routes/templates/n1_templates_1_0_0.json'))
# n5_3_templates = json.load(open('../teamdrive/projects/n5routes/templates/n5-3_templates_1_0_0.json'))
all_templates = json.load(open('../teamdrive/projects/n5routes/templates/all_routes_templates_1_0_0.json'))
all_templates_0 = json.load(open('../teamdrive/projects/n5routes/templates/all_routes_templates_0_0_0.json'))
# n5_3_templates_0 = json.load(open('../teamdrive/projects/n5routes/templates/n5-3_templates_0_0_0.json'))
# n5_3_mcts_templates_0 = json.load(open('../teamdrive/projects/n5routes/templates/n5-3_mcts_templates_0_0_0.json'))
golden_dict = pickle.load(open('../teamdrive/projects/n5routes/templates/golden_dict.pickle', 'rb'))
print(len(all_routes), len(all_templates))

def extract_mol_set(route, instock=False):
    mol_set = set()
    def preorder(root):
        if root.get('children') == None:
            return
        for child in root['children']:
            if child.get('type') == 'mol':
                if instock:
                    if not child['in_stock']:
                        mol_set.add(child['smiles'])
                else:
                    mol_set.add(child['smiles'])
            preorder(child)
    preorder(route)
    return mol_set
def extract_mol(route):
    mol_nodes = [route]
    def preorder(root):
        if root.get('children') == None:
            return
        for child in root['children']:
            if child.get('type') == 'mol':
                mol_nodes.append(child)
            preorder(child)
    preorder(route)
    return mol_nodes
def extract_rxn(route):
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
def extract_bigram(rxn_nodes):
    digram_rxns, digram_templates = [], []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles']) in digram_rxns:
                digram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles']))
                try:
                    digram_templates.append((rxn_node['metadata']['template'], child_rxn['metadata']['template']))
                except:
                    pass
    return digram_rxns, digram_templates
def extract_trigram(rxn_nodes):
    trigram_rxns, trigram_templates = [], []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            child_child_rxns = [child['children'][0] for child in child_rxn['children'] if child.get('children') is not None]
            for child_child_rxn in child_child_rxns:
                if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles']) in trigram_rxns:
                    trigram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles']))
                    try:
                        trigram_templates.append((rxn_node['metadata']['template'], child_rxn['metadata']['template'], child_child_rxn['metadata']['template']))
                    except:
                        pass
    return trigram_rxns, trigram_templates
def extract_tetragram(rxn_nodes):
    tetragram_rxns, tetragram_templates = [], []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            child_child_rxns = [child['children'][0] for child in child_rxn['children'] if child.get('children') is not None]
            for child_child_rxn in child_child_rxns:
                child_child_child_rxns = [child['children'][0] for child in child_child_rxn['children'] if child.get('children') is not None]
                for child_child_child_rxn in child_child_child_rxns:
                    if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles'], child_child_child_rxn['metadata']['smiles']) in tetragram_rxns:
                        tetragram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles'], child_child_child_rxn['metadata']['smiles']))
                        try:
                            tetragram_templates.append((rxn_node['metadata']['template'], child_rxn['metadata']['template'], child_child_rxn['metadata']['template'], child_child_child_rxn['metadata']['template']))
                        except:
                            pass
    return tetragram_rxns, tetragram_templates
def extract_pentagram(rxn_nodes):
    pentagram_rxns, pentagram_templates = [], []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            child_child_rxns = [child['children'][0] for child in child_rxn['children'] if child.get('children') is not None]
            for child_child_rxn in child_child_rxns:
                child_child_child_rxns = [child['children'][0] for child in child_child_rxn['children'] if child.get('children') is not None]
                for child_child_child_rxn in child_child_child_rxns:
                    child_child_child_child_rxns = [child['children'][0] for child in child_child_child_rxn['children'] if child.get('children') is not None]
                    for child_child_child_child_rxn in child_child_child_child_rxns:
                        if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles'], child_child_child_rxn['metadata']['smiles'], child_child_child_child_rxn['metadata']['smiles']) in pentagram_rxns:
                            pentagram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles'], child_child_child_rxn['metadata']['smiles'], child_child_child_child_rxn['metadata']['smiles']))
                            try:
                                pentagram_templates.append((rxn_node['metadata']['template'], child_rxn['metadata']['template'], child_child_rxn['metadata']['template'], child_child_child_rxn['metadata']['template'], child_child_child_child_rxn['metadata']['template']))
                            except:
                                pass
                            
    return pentagram_rxns, pentagram_templates
def extract_generated_bigram(rxn_nodes):
    bigram_rxns, bigram_templates = [], []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            if not (rxn_node['metadata']['mapped_reaction_smiles'], child_rxn['metadata']['mapped_reaction_smiles']) in bigram_rxns:
                bigram_rxns.append((rxn_node['metadata']['mapped_reaction_smiles'], child_rxn['metadata']['mapped_reaction_smiles']))
                try:
                    bigram_templates.append((rxn_node['metadata']['template'], child_rxn['metadata']['template']))
                except:
                    pass
    return bigram_rxns, bigram_templates

def route_score(
    tree_dict,
    mol_costs=None,
    average_yield=0.8,
    reaction_cost=1.0,
) -> float:
    """
    Calculate the score of route using the method from
    (Badowski et al. Chem Sci. 2019, 10, 4640).

    The reaction cost is constant and the yield is an average yield.
    The starting materials are assigned a cost based on whether they are in
    stock or not. By default starting material in stock is assigned a
    cost of 1 and starting material not in stock is assigned a cost of 10.

    To be accurate, each molecule node need to have an extra
    boolean property called `in_stock`.

    :param tree_dict: the route to analyze
    :param mol_costs: the starting material cost
    :param average_yield: the average yield, defaults to 0.8
    :param reaction_cost: the reaction cost, defaults to 1.0
    :return: the computed cost
    """
    mol_cost = mol_costs or {True: 1, False: 10}

    reactions = tree_dict.get("children", [])
    if not reactions:
        return mol_cost[tree_dict.get("in_stock", True)]

    child_sum = sum(
        1 / average_yield * route_score(child) for child in reactions[0]["children"]
    )
    return reaction_cost + child_sum

def route_score_bleu_2(routes, vocab_negative, vocab_positive, golden_template):
    min_length = min([len(extract_rxn(r)) for r in routes])
    scores = []
    for route in routes:
        rxn_nodes = extract_rxn(route)
        if len(rxn_nodes) == 0:
            scores.append(0)
            continue
        try:
            bigrams = extract_generated_bigram(rxn_nodes)[1]
        except:
            bigrams_orig = extract_bigram(rxn_nodes)[0]
            bigrams = [tuple([golden_template[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in bigrams_orig]
        if len(bigrams) == 0:
            bigram_ratio = 0
            bigram_freq = 0
        else:
            bigram_ratio = sum([b in vocab_positive for b in bigrams]) / len(bigrams)
            bigram_freq = sum([vocab_positive[b] for b in bigrams if b in vocab_positive])
        if len(bigrams) == 0:
            bigram_ratio_neg = 0
        else:
            bigram_ratio_neg = sum([b in vocab_negative for b in bigrams]) / len(bigrams)
        scores.append(- 1 * np.exp(bigram_ratio) - 1 * np.exp(min_length / len(rxn_nodes)) + 1 * np.exp(bigram_ratio_neg) - 1e-5 * bigram_freq)
    return scores


def route_score_probability(route):
    rxn_nodes = extract_rxn(route)
    if len(rxn_nodes) == 0:
        return 0
    cumul_prob = 1
    for rxn_node in rxn_nodes:
        cumul_prob *= rxn_node['metadata']['policy_probability']
    return cumul_prob

def route_score_probability_golden(route, golden_dict):
    mols = extract_mol(route)
    mols = [mol for mol in mols if not mol['in_stock']]
    cumul_prob = 1
    for mol in mols:
        single_step_result = golden_dict[mol['smiles']]
        candidate_reactants = [set(reactant[0].reaction_smiles().split('>>')[-1].split('.')) for reactant in single_step_result]
        golden_set = set([reactant['smiles'] for reactant in mol['children'][0]['children']])
        if golden_set in candidate_reactants:
            cumul_prob *= single_step_result[candidate_reactants.index(golden_set)][0].metadata['policy_probability']
        else:
            if len(single_step_result):
                cumul_prob *= 1e-10#single_step_result[-1][0].metadata['policy_probability']
            else:
                cumul_prob *= 1e-10
    return cumul_prob

def calc_acc(route_type, route_name, add_golden=True, neg_bigram=set()):
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
        pos_bigram = pickle.load(open('data/pmi_bigrams_n5.pkl', 'rb'))
        golden_template = n5_templates
    else:
        golden_routes = n1_routes
        pos_bigram = pickle.load(open('data/pmi_bigrams_n1.pkl', 'rb'))
        golden_template = n1_templates
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
            candidate_bleu = route_score_bleu_2([golden_route] + routes, neg_bigram, pos_bigram, golden_template)
            golden_bleu = candidate_bleu[0]
            candidate_bleu = candidate_bleu[1:]
            golden_prob = route_score_probability_golden(golden_route, golden_dict)
            candidate_probs = [route_score_probability(r) for r in routes]
        else:
            if golden_exist:
                golden_idx = candidate_sets.index(golden_set)
            else:
                continue
            candidate_scores = [route_score(r) for r in routes]
            candidate_bleu = route_score_bleu_2(routes, neg_bigram, pos_bigram, golden_template)
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
# calc_acc('n5routes', 'mcts_v2', add_golden=False)
'''calc_acc('n5routes', 'exclude_neg_1_3')
calc_acc('n5routes', 'exclude_neg_1_3', add_golden=False)
calc_acc('n5routes', 'exclude_neg_1_5')
calc_acc('n5routes', 'exclude_neg_1_5', add_golden=False)
calc_acc('n5routes', 'exclude_neg_1_8')
calc_acc('n5routes', 'exclude_neg_1_8', add_golden=False)
calc_acc('n5routes', 'exclude_neg_3_5')
calc_acc('n5routes', 'exclude_neg_3_5', add_golden=False)
calc_acc('n5routes', 'exclude_neg_5_5')
calc_acc('n5routes', 'exclude_neg_5_5', add_golden=False)'''
calc_acc('n1routes', 'mcts_v2')
calc_acc('n5routes', 'retrostar_v2')
calc_acc('n1routes', 'retrostar_v2')
# calc_acc('n1routes', 'mcts_v2', add_golden=False)
'''calc_acc('n1routes', 'exclude_neg_1_3')
calc_acc('n1routes', 'exclude_neg_1_3', add_golden=False)
calc_acc('n1routes', 'exclude_neg_1_5')
calc_acc('n1routes', 'exclude_neg_1_5', add_golden=False)
calc_acc('n1routes', 'exclude_neg_1_8')
calc_acc('n1routes', 'exclude_neg_1_8', add_golden=False)
calc_acc('n1routes', 'exclude_neg_3_5')
calc_acc('n1routes', 'exclude_neg_3_5', add_golden=False)
calc_acc('n1routes', 'exclude_neg_5_5')
calc_acc('n1routes', 'exclude_neg_5_5', add_golden=False)
calc_acc('n5routes', 'retrostar_v2')
calc_acc('n5routes', 'retrostar_v2', add_golden=False)
calc_acc('n1routes', 'retrostar_v2')
calc_acc('n1routes', 'retrostar_v2', add_golden=False)'''