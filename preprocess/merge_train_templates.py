import json
import pickle
import os

train_routes = pickle.load(open('../data/routes_train.pkl', 'rb'))
train_template_0 = {}
train_template_1 = {}
train_template_2 = {}
for i in range(20):
    train_template_0.update(json.load(open(f'../data/{i}_train_routes_templates_0_0_0.json')))
    train_template_1.update(json.load(open(f'../data/{i}_train_routes_templates_1_0_0.json')))
    train_template_2.update(json.load(open(f'../data/{i}_train_routes_templates_2_0_0.json')))

for rxn in set([i for j in train_routes for i in j]):
    if rxn not in train_template_0:
        train_template_0[rxn] = rxn
    if rxn not in train_template_1:
        train_template_1[rxn] = rxn
    if rxn not in train_template_2:
        train_template_2[rxn] = rxn

json.dump(train_template_0, open('../data/train_routes_templates_0_0_0.json', 'w'))
json.dump(train_template_1, open('../data/train_routes_templates_1_0_0.json', 'w'))
json.dump(train_template_2, open('../data/train_routes_templates_2_0_0.json', 'w'))

for i in range(20):
    os.remove(f'../data/{i}_train_routes_templates_0_0_0.json')
    os.remove(f'../data/{i}_train_routes_templates_1_0_0.json')
    os.remove(f'../data/{i}_train_routes_templates_2_0_0.json')