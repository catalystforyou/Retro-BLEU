import pickle

all_routes = []
routes = pickle.load(open('../PaRoutes/setup/loaded_routes.pickle', 'rb'))
for k, v in routes.items():
    for route in v:
        all_routes.append(route['rt'])
pickle.dump(all_routes, open('../data/all_routes.pickle', 'wb'))