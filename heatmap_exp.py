from resnet_cifar_experiments import load_data, train_network, mse_loss
from mlp import CentralMLP

import argparse
import numpy as np

def IST_Converge(X, Y, num_neurons, num_subnets, init_scale, prob, 
        learning_rate, local_iter, window_size, mask_type, verbose=0):
    
    full_model = CentralMLP(num_neurons, X.shape[1], 10, init_scale, prob)
    error_hist = []
    error_hist.append(mse_loss(full_model.forward(X), Y).numpy())
    if verbose > 0:
        print('The Error of Iteration 0 is %f' % error_hist[-1])
    patience = 0
    best_avg = float('inf')
    gstep = 0
    while True:
        
        gstep += 1
        subnets, masks = full_model.generate_subnets(num_subnets, method=mask_type)
        for _, subnet in enumerate(subnets):
            train_network(subnet, X, Y, learning_rate, local_iter)
        
        full_model.aggregate_updates(subnets, masks)
        error_hist.append(mse_loss(full_model.forward(X), Y).numpy())
        avg = np.mean(error_hist[np.maximum(0, gstep - window_size):])
        if avg < best_avg:
            best_avg = avg
            patience = 0
        else:
            patience += 1
        
        if verbose > 0 and (gstep + 1) % verbose == 0:
            print('The Avg Error of Iteration %d is %f' % (gstep + 1, avg))
        
        if patience > 0 or best_avg < 5e-4:
            break
        
    return best_avg

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--num_neurons', type=int, default=100)
    parser.add_argument('-n', '--num_samples', type=int, default=100)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-e', '--local_iter', type=int, default=1)
    parser.add_argument('-k', '--init_scale', type=float, default=1.)
    parser.add_argument('-w', '--window_size', type=int, default=10)
    parser.add_argument('-t', '--mask_type', type=str, choices=['Bernoulli', 'Categorical'], default='Bernoulli')
    parser.add_argument('-r', '--repetition', type=int, default=1)
    parser.add_argument('-f', '--result_fname', type=str, default='result.txt')
    args = parser.parse_args()
    
    X, Y = load_data(args.num_samples)
    weight = np.random.normal(scale=1, size=(X.shape[1], args.num_neurons))
    a = np.random.choice([-1, 1], size=(args.num_neurons,), p=[0.5, 0.5])
    
    if args.mask_type == 'Bernoulli':
        
        workers = np.arange(1, 11)
        xis = np.arange(1, 11)
        local_iter = 5
        heatmap_data = np.zeros((10, 10))
        for w in workers:
            for x in xis:
                cur_w = 1 if x == 10 else w
                best_avg = 0.
                for _ in range(args.repetition):
                    best_avg += IST_Converge(X, Y, args.num_neurons, cur_w, 
                                             args.init_scale, x / 10, 
                                             args.learning_rate, local_iter,
                                             args.window_size, args.mask_type, verbose=0)
                heatmap_data[w - 1, x - 1] = best_avg / args.repetition
                print('Worker %d, Prob %f: %f' % (w, x / 10, heatmap_data[w - 1, x - 1]))
        np.savetxt(args.result_fname, heatmap_data)   
