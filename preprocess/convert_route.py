import pickle

routes = pickle.load(open('../PaRoutes/setup/loaded_routes.pickle', 'rb'))
pickle.dump(routes['rt'], open('../data/all_routes.pickle', 'wb'))