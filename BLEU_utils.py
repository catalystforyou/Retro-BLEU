import numpy as np
from rxnmapper import RXNMapper
from rxnutils.chem.reaction import ChemicalReaction

def get_template(mapped_rxn, radius=1):
    try:
        template = ChemicalReaction(mapped_rxn).generate_reaction_template(radius)[1].smarts
    except:
        template = ''
    return template

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

def extract_mols(route):
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

def extract_bigram(rxn_nodes):
    bigram_rxns = []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles']) in bigram_rxns:
                bigram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles']))
    return bigram_rxns

def extract_trigram(rxn_nodes):
    trigram_rxns = []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            child_child_rxns = [child['children'][0] for child in child_rxn['children'] if child.get('children') is not None]
            for child_child_rxn in child_child_rxns:
                if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles']) in trigram_rxns:
                    trigram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles']))
    return trigram_rxns

def extract_tetragram(rxn_nodes):
    tetragram_rxns = []
    for rxn_node in rxn_nodes:
        child_rxns = [child['children'][0] for child in rxn_node['children'] if child.get('children') is not None]
        for child_rxn in child_rxns:
            child_child_rxns = [child['children'][0] for child in child_rxn['children'] if child.get('children') is not None]
            for child_child_rxn in child_child_rxns:
                child_child_child_rxns = [child['children'][0] for child in child_child_rxn['children'] if child.get('children') is not None]
                for child_child_child_rxn in child_child_child_rxns:
                    if not (rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles'], child_child_child_rxn['metadata']['smiles']) in tetragram_rxns:
                        tetragram_rxns.append((rxn_node['metadata']['smiles'], child_rxn['metadata']['smiles'], child_child_rxn['metadata']['smiles'], child_child_child_rxn['metadata']['smiles']))
    return tetragram_rxns

def extract_pentagram(rxn_nodes):
    pentagram_rxns = []
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
    return pentagram_rxns

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

def extract_rxns_askcos(route):
    mol_nodes = [route]
    rxns = []
    def preorder(root):
        if len(root['child']) == 0:
            return
        for child in root['child']:
            mol_nodes.append(child)
            preorder(child)
    preorder(route)
    for mol_node in mol_nodes:
        if len(mol_node['child']) != 0:
            rxns.append(mol_node['smiles'] + '>>' + '.'.join([m['smiles'] for m in mol_node['child']]))
    return rxns


def build_vocab(routes, template_dict):
    all_ngram_rxns = [[] for _ in range(4)]
    all_rxns = []
    for idx, route in enumerate(routes):
        rxn_nodes = extract_rxns(route)
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

def evaluate_routes(test_routes, template_dict, vocab_ngram_templates, vocab_ngram_rxns):
    bleu_score = [[] for _ in range(4)]
    bleu_template_score = [[] for _ in range(4)]
    for route in test_routes:
        try:
            rxn_nodes = extract_rxns(route)
            bigram, trigram, tetragram, pentagram = extract_bigram(rxn_nodes), extract_trigram(rxn_nodes), extract_tetragram(rxn_nodes), extract_pentagram(rxn_nodes)
            if len(bigram) > 0:
                bleu_score[0].append(sum([bi in vocab_ngram_rxns[0] for bi in bigram]) / len(bigram))
                bigram_template = [tuple([template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in bigram]
                bleu_template_score[0].append(sum([bi in vocab_ngram_templates[0] for bi in bigram_template]) / len(bigram_template))
            if len(trigram) > 0:
                bleu_score[1].append(sum([tri in vocab_ngram_rxns[1] for tri in trigram]) / len(trigram))
                trigram_template = [tuple([template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in trigram]
                bleu_template_score[1].append(sum([tri in vocab_ngram_templates[1] for tri in trigram_template]) / len(trigram_template))
            if len(tetragram) > 0:
                bleu_score[2].append(sum([tetra in vocab_ngram_rxns[2] for tetra in tetragram]) / len(tetragram))
                tetragram_template = [tuple([template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in tetragram]
                bleu_template_score[2].append(sum([tetra in vocab_ngram_templates[2] for tetra in tetragram_template]) / len(tetragram_template))
            if len(pentagram) > 0:
                bleu_score[3].append(sum([penta in vocab_ngram_rxns[3] for penta in pentagram]) / len(pentagram))
                pentagram_template = [tuple([template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in pentagram]
                bleu_template_score[3].append(sum([penta in vocab_ngram_templates[3] for penta in pentagram_template]) / len(pentagram_template))
        except:
            test_ngram_rxns, test_ngram_templates = get_test_ngram(route, template_dict)
            if len(test_ngram_rxns[0]) > 0:
                bleu_score[0].append(sum([bi in vocab_ngram_rxns[0] for bi in test_ngram_rxns[0]]) / len(test_ngram_rxns[0]))
                bleu_template_score[0].append(sum([bi in vocab_ngram_templates[0] for bi in test_ngram_templates[0]]) / len(test_ngram_templates[0]))
            if len(test_ngram_rxns[1]) > 0:
                bleu_score[1].append(sum([tri in vocab_ngram_rxns[1] for tri in test_ngram_rxns[1]]) / len(test_ngram_rxns[1]))
                bleu_template_score[1].append(sum([tri in vocab_ngram_templates[1] for tri in test_ngram_templates[1]]) / len(test_ngram_templates[1]))
            if len(test_ngram_rxns[2]) > 0:
                bleu_score[2].append(sum([tetra in vocab_ngram_rxns[2] for tetra in test_ngram_rxns[2]]) / len(test_ngram_rxns[2]))
                bleu_template_score[2].append(sum([tetra in vocab_ngram_templates[2] for tetra in test_ngram_templates[2]]) / len(test_ngram_templates[2]))
            if len(test_ngram_rxns[3]) > 0:
                bleu_score[3].append(sum([penta in vocab_ngram_rxns[3] for penta in test_ngram_rxns[3]]) / len(test_ngram_rxns[3]))
                bleu_template_score[3].append(sum([penta in vocab_ngram_templates[3] for penta in test_ngram_templates[3]]) / len(test_ngram_templates[3]))
    return bleu_score, bleu_template_score

def get_new_ngram(route, template_radius=1):
    rxnmapper = RXNMapper()
    all_ngram_rxns = [[] for _ in range(4)]
    precursors_dict = {rxn.split('>>')[0]:rxn for rxn in route}
    forward_rxns = ['>>'.join([rxn.split('>')[-1], rxn.split('>')[0]]) for rxn in route]
    mapped_rxns = rxnmapper.get_attention_guided_atom_maps(forward_rxns)
    mapped_rxns = [rxn['mapped_rxn'] for rxn in mapped_rxns]
    test_template_dict = {rxn:get_template(mapped_rxn, radius=template_radius) for (rxn, mapped_rxn) in zip(route, mapped_rxns)}
    for rxn in route:
        reactants = rxn.split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[0].append((rxn, precursors_dict[reactant]))
    for gram in all_ngram_rxns[0]:
        reactants = gram[-1].split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[1].append(gram + tuple([precursors_dict[reactant]]))
    for gram in all_ngram_rxns[1]:
        reactants = gram[-1].split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[2].append(gram + tuple([precursors_dict[reactant]]))
    for gram in all_ngram_rxns[2]:
        reactants = gram[-1].split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[3].append(gram + tuple([precursors_dict[reactant]]))
    all_ngram_templates = [[tuple([test_template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in n_gram_rxns] for n_gram_rxns in all_ngram_rxns]
    return all_ngram_rxns, all_ngram_templates

def get_test_ngram(route, template_dict):
    all_ngram_rxns = [[] for _ in range(4)]
    precursors_dict = {rxn.split('>>')[0]:rxn for rxn in route}
    for rxn in route:
        reactants = rxn.split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[0].append((rxn, precursors_dict[reactant]))
    for gram in all_ngram_rxns[0]:
        reactants = gram[-1].split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[1].append(gram + tuple([precursors_dict[reactant]]))
    for gram in all_ngram_rxns[1]:
        reactants = gram[-1].split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[2].append(gram + tuple([precursors_dict[reactant]]))
    for gram in all_ngram_rxns[2]:
        reactants = gram[-1].split('>>')[1].split('.')
        for reactant in reactants:
            if reactant in precursors_dict:
                all_ngram_rxns[3].append(gram + tuple([precursors_dict[reactant]]))
    all_ngram_templates = [[tuple([template_dict[rxn] for rxn in curr_rxn_set]) for curr_rxn_set in n_gram_rxns] for n_gram_rxns in all_ngram_rxns]
    return all_ngram_rxns, all_ngram_templates

def build_route_dict(routes):
    route_dict = {}
    for route in routes:
        product = route[0].split('>>')[0]
        route_dict[product] = route
    return route_dict

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

def heuristic_score(length):
    if length <= 3:
        return 1
    else:
        return 3 / length

def route_score_bleu(routes, vocab_negative, vocab_positive, golden_template, heuristic=True):
    min_length = min([len(extract_rxns(r)) for r in routes])
    scores = []
    for route in routes:
        rxn_nodes = extract_rxns(route)
        if len(rxn_nodes) == 0:
            scores.append(0)
            continue
        try:
            bigrams = extract_generated_bigram(rxn_nodes)[1]
        except:
            bigrams_orig = extract_bigram(rxn_nodes)
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
        if heuristic:
            scores.append(- 1 * np.exp(bigram_ratio) - 1 * np.exp(heuristic_score(len(rxn_nodes))) + 1 * np.exp(bigram_ratio_neg))
        else:
            scores.append(- 1 * np.exp(bigram_ratio) - 1 * np.exp(min_length / len(rxn_nodes)) + 1 * np.exp(bigram_ratio_neg))
    return scores

def retrostar_route_bleu(routes, vocab_negative, vocab_positive, golden_template, heuristic=True):
    min_length = min([len(route) for route in routes])
    scores = []
    for route in routes:
        ngram_rxns, ngram_templates = get_test_ngram(route, golden_template)
        if len(ngram_templates[0]) == 0:
            bigram_ratio = 0
            bigram_ratio_neg = 0
        else:
            bigram_ratio = sum([b in vocab_positive for b in ngram_templates[0]]) / len(ngram_templates[0])
            bigram_ratio_neg = sum([b in vocab_negative for b in ngram_templates[0]]) / len(ngram_templates[0])
        if heuristic:
            scores.append(- 1 * np.exp(bigram_ratio) - 1 * np.exp(heuristic_score(len(route))) + 1 * np.exp(bigram_ratio_neg))
        else:
            scores.append(- 1 * np.exp(bigram_ratio) - 1 * np.exp(min_length / len(route)) + 1 * np.exp(bigram_ratio_neg))
    return scores

def route_score_probability(route):
    rxn_nodes = extract_rxns(route)
    if len(rxn_nodes) == 0:
        return 0
    cumul_prob = 1
    for rxn_node in rxn_nodes:
        cumul_prob *= rxn_node['metadata']['policy_probability']
    return cumul_prob

def route_score_probability_golden(route, golden_dict, searchable=True):
    mols = extract_mols(route)
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
                cumul_prob *= 0 if searchable else 1e-10
            else:
                cumul_prob *= 0 if searchable else 1e-10
    return cumul_prob