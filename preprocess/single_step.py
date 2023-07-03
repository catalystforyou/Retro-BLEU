from aizynthfinder.aizynthfinder import AiZynthExpander
expander = AiZynthExpander(
    configdict={
        "policy": {
            "files": {
                "paroutes": [
                    "../data/uspto_keras_model.hdf5", 
                    "../data/uspto_unique_templates.csv.gz"
                ]
            }
        }
    }
)
expander.expansion_policy.select("paroutes")

import json
import pickle
n1_routes = json.load(open('../data/n1-routes.json'))
n5_routes = json.load(open('../data/n5-routes.json'))
def extract_mol_set(route):
    mol_set = set()
    mol_set.add(route['smiles'])
    def preorder(root):
        if root.get('children') == None:
            return
        for child in root['children']:
            if child.get('type') == 'mol':
                if not child['in_stock']:
                    mol_set.add(child['smiles'])
            preorder(child)
    preorder(route)
    return mol_set

n1_nodes = set()
for route in n1_routes:
    n1_nodes.update(extract_mol_set(route))
n5_nodes = set()
for route in n5_routes:
    n5_nodes.update(extract_mol_set(route))
print(len(n1_nodes))
print(len(n5_nodes))

all_nodes = n1_nodes.union(n5_nodes)
print(len(all_nodes))

golden_dict = {}
from tqdm import tqdm
for smi in tqdm(all_nodes):
    # smi = smi['smiles']
    predictions = expander.do_expansion(smi, return_n=50)
    cum_prob = 0
    for idx, pred in enumerate(predictions):
        cum_prob += pred[0].metadata['policy_probability']
        if cum_prob >= 0.995:
            predictions = predictions[:idx+1]
            break
    golden_dict[smi] = predictions

pickle.dump(golden_dict, open('../data/golden_dict.pickle', 'wb'))