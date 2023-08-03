import json
from tqdm import trange

n1_mcts_routes = []
n1_mcts_routes_10 = []
for i in trange(10000):
    curr_route = json.load(open('../routes/n1routes/mcts_v2/routes_{}.json'.format(i)))
    if len(curr_route) > 10:
        n1_mcts_routes_10.append(curr_route[:10])
    else:
        n1_mcts_routes_10.append(curr_route)
    n1_mcts_routes.append(json.load(open('../routes/n1routes/mcts_v2/routes_{}.json'.format(i)))[0])

n5_mcts_routes = []
n5_mcts_routes_10 = []
for i in trange(10000):
    n5_mcts_routes.append(json.load(open('../routes/n5routes/mcts_v2/routes_{}.json'.format(i)))[0])
    curr_route = json.load(open('../routes/n5routes/mcts_v2/routes_{}.json'.format(i)))
    if len(curr_route) > 10:
        n5_mcts_routes_10.append(curr_route[:10])
    else:
        n5_mcts_routes_10.append(curr_route)

n1_retrostar_routes = []
n1_retrostar_routes_10 = []
for i in trange(10000):
    n1_retrostar_routes.append(json.load(open('../routes/n1routes/retrostar_v2/routes_{}.json'.format(i)))[0])
    curr_route = json.load(open('../routes/n1routes/retrostar_v2/routes_{}.json'.format(i)))
    if len(curr_route) > 10:
        n1_retrostar_routes_10.append(curr_route[:10])
    else:
        n1_retrostar_routes_10.append(curr_route)

n5_retrostar_routes = []
n5_retrostar_routes_10 = []
for i in trange(10000):
    n5_retrostar_routes.append(json.load(open('../routes/n5routes/retrostar_v2/routes_{}.json'.format(i)))[0])
    curr_route = json.load(open('../routes/n5routes/retrostar_v2/routes_{}.json'.format(i)))
    if len(curr_route) > 10:
        n5_retrostar_routes_10.append(curr_route[:10])
    else:
        n5_retrostar_routes_10.append(curr_route)

json.dump(n1_mcts_routes, open('data/n1_mcts_firstroutes.json', 'w'))
json.dump(n5_mcts_routes, open('data/n5_mcts_firstroutes.json', 'w'))
json.dump(n1_retrostar_routes, open('data/n1_retrostar_firstroutes.json', 'w'))
json.dump(n5_retrostar_routes, open('data/n5_retrostar_firstroutes.json', 'w'))
json.dump(n1_mcts_routes_10, open('data/n1_mcts_firstroutes_10.json', 'w'))
json.dump(n5_mcts_routes_10, open('data/n5_mcts_firstroutes_10.json', 'w'))
json.dump(n1_retrostar_routes_10, open('data/n1_retrostar_firstroutes_10.json', 'w'))
json.dump(n5_retrostar_routes_10, open('data/n5_retrostar_firstroutes_10.json', 'w'))