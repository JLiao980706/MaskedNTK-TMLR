import tensorflow as tf
import numpy as np


class SubMLP:
    
    def __init__(self, W, a, scaling_factor):
        self.W = tf.Variable(W, dtype=tf.float32)
        self.a = a
        self.scaling_factor = scaling_factor
    
    def forward(self, X):
        return tf.matmul(tf.nn.relu(tf.matmul(X, self.W)), self.a) * \
            self.scaling_factor
    
    def get_weight(self):
        return self.W
    
    def get_weight_val(self):
        return self.W.numpy()


class CentralMLP:
    
    def __init__(self, m, d, dout, kappa, prob):
        self.m = m
        self.d = d
        self.kappa = kappa
        self.prob = prob
        # self.W = tf.keras.initializers.GlorotUniform()(shape=(d, m)).numpy()
        self.W = np.random.normal(size=(d, m)) / np.sqrt(m)
        self.a = np.random.choice([-1, 1], size=(m, dout), p=[0.5, 0.5])
        self.scaling_factor = 1 / np.sqrt(dout)
                
    def forward(self, X):
        return self.scaling_factor * \
            tf.matmul(tf.nn.relu(tf.matmul(X, self.W)), self.a)
    
    def generate_subnets(self, num_nets, method='Bernoulli'):
        
        subnets = []
        if method == 'Bernoulli':
            masks = np.random.choice([0, 1], size=(num_nets, self.m), p=[1 - self.prob, self.prob])
            
        elif method == 'Categorical':
            idxs = np.random.choice(num_nets, size=(self.m,))
            masks = np.zeros((num_nets, self.m))
            
            for l in range(num_nets):
                masks[l, idxs==l] = 1
        
        else:
            raise Exception('Masking method not recognized.')
        
        for l in range(num_nets):
            subnet_Ws = self.W[:, masks[l] == 1]
            subnet_as = self.a[masks[l] == 1, :]
            subnets.append(SubMLP(subnet_Ws, subnet_as, self.scaling_factor / self.prob))
        
        return subnets, masks
    
    def aggregate_updates(self, subnets, masks):
        
        self.W = np.zeros_like(self.W)
        total_workers = np.sum(masks, axis=0)
        global_stepsize = np.where(total_workers == 0, 0, 1. / total_workers).reshape((1, -1))
        
        for l in range(len(subnets)):
            local_W = np.zeros_like(self.W)
            local_W[:, masks[l] == 1] = subnets[l].get_weight_val()
            self.W += global_stepsize * local_W
