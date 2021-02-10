import numpy as np


class NiceDummy :
    def __init__(self, num_coupling_layers, path_lenght):
        #Dans cette version dummy seul path_lenght compte
        self.coupling_layers = None
        self.num_coupling_layers = num_coupling_layers
        self.path_length = path_lenght

    def learn(self, paths, proba):
        if paths.shape[1] != 2*self.path_length :
            print("Wrong path size")

        if len(proba) != paths.shape[0] :
            print("Paths and probas not matching")

    def generate_paths(self, num_path):
        print("generate_paths")
        paths = np.random.uniform(size=(num_path, 2*self.path_length))
        probas = np.ones((num_path))

        return paths, probas

