import numpy as np
import tensorflow as tf
import argparse
import json

from mlp import CentralMLP, SubMLP


def load_data(num_samples):
    
    regression_data = np.load('./data/resnet50_data.npz')
    X = regression_data['features']
    Y = regression_data['labels']
    X /= np.linalg.norm(X, axis=1).reshape((-1, 1))
    Y /= np.abs(Y).max()
    if num_samples > X.shape[0]:
        num_samples = X.shape[0]
    
    return X[:num_samples], Y[:num_samples]


def mse_loss(y_pred, y_true):
    result = 0.5 * tf.reduce_sum(tf.math.pow(y_pred - y_true, 2)) / y_true.shape[0]
    return result

def train_network(network, X, Y, learning_rate, iter):
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    network_weight = network.get_weight()
    
    error_hist = np.zeros(iter)
    for i in range(iter):
        
        with tf.GradientTape() as gtape:
            gtape.watch(network_weight)
            loss = mse_loss(network.forward(X), Y)
            
        gradient = gtape.gradient(loss, network_weight)
        opt.apply_gradients([(gradient, network_weight)])
        error_hist[i] = loss.numpy()
    return error_hist
        
        
def IST(X, Y, num_neurons, num_subnets, init_scale, prob, 
        learning_rate, local_iter, global_iter, mask_type, verbose=0):
    
    full_model = CentralMLP(num_neurons, X.shape[1], 10, init_scale, prob)
    error_hist = []
    error_hist.append(mse_loss(full_model.forward(X), Y).numpy())
    print('The Error of Iteration 0 is %f' % error_hist[-1])
    
    local_hist_mean = np.zeros((global_iter, local_iter))
    for gstep in range(global_iter):
        
        subnets, masks = full_model.generate_subnets(num_subnets, method=mask_type)
        hists = np.zeros((num_subnets, local_iter))
        for net_idx, subnet in enumerate(subnets):
            hists[net_idx] = train_network(subnet, X, Y, learning_rate, local_iter)
            
        local_hist_mean[gstep] = hists.mean(axis=0)
        
        full_model.aggregate_updates(subnets, masks)
        error_hist.append(mse_loss(full_model.forward(X), Y).numpy())
        
        if verbose > 0 and (gstep + 1) % verbose == 0:
            print('The Error of Iteration %d is %f' % (gstep + 1, error_hist[-1]))
        
    return error_hist, local_hist_mean


def IST_Converge(X, Y, num_neurons, num_subnets, init_scale, prob, 
        learning_rate, local_iter, mask_type, verbose=0):
    
    full_model = CentralMLP(num_neurons, X.shape[1], 10, init_scale, prob)
    error_hist = []
    error_hist.append(mse_loss(full_model.forward(X), Y).numpy())
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
        avg = np.mean(error_hist[np.max([0, gstep - 10]):])
        if avg < best_avg:
            best_avg = avg
        else:
            patience += 1
        
        if verbose > 0 and (gstep + 1) % verbose == 0:
            print('The Avg Error of Iteration %d is %f' % (gstep + 1, avg))
        
        if patience > 5:
            break
        
    return best_avg


def IST_try(X, Y, num_neurons, num_subnets, init_scale, prob, 
        learning_rate, local_iter, mask_type):
    
    while True:
        
        full_model = CentralMLP(num_neurons, X.shape[1], 10, init_scale, prob)
        init_error = mse_loss(full_model.forward(X), Y).numpy()
        subnets, masks = full_model.generate_subnets(num_subnets, method=mask_type)
        
        for subnet in subnets:
            _ = train_network(subnet, X, Y, learning_rate, local_iter)
        
        full_model.aggregate_updates(subnets, masks)
        iter_error = mse_loss(full_model.forward(X), Y).numpy()
        print('The Error of Learning Rate %f is %f' % (learning_rate, iter_error))
        
        if iter_error <= init_error:
            break
        
        learning_rate /= 2
        
    return learning_rate


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['try', 'run'])
    parser.add_argument('-m', '--num_neurons', type=int, default=100)
    parser.add_argument('-n', '--num_samples', type=int, default=100)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-p', '--num_workers', type=int, default=1)
    parser.add_argument('-x', '--probability', type=float, default=1.)
    parser.add_argument('-e', '--local_iter', type=int, default=1)
    parser.add_argument('-g', '--global_iter', type=int, default=100)
    parser.add_argument('-k', '--init_scale', type=float, default=1.)
    parser.add_argument('-t', '--mask_type', type=str, choices=['Bernoulli', 'Categorical'], default='Bernoulli')
    parser.add_argument('-r', '--repetition', type=int, default=1)
    parser.add_argument('-f', '--result_fname', type=str, default='result.txt')
    parser.add_argument('-v', '--verbose', type=int, default=0)
    
    args = parser.parse_args()
    info_dict = vars(args)
    X, Y = load_data(args.num_samples)
    
    if args.mode == 'try':
        lr = IST_try(X, Y, args.num_neurons, args.num_workers, args.init_scale, 
                     args.probability, args.learning_rate, args.local_iter, 
                     args.mask_type)
        print('The best learning rate is %f' % lr)
        exit()
    
    error_hists = np.zeros((args.repetition, args.global_iter + 1))
    local_error_hists = np.zeros((args.repetition, args.global_iter, args.local_iter))
    for i in range(args.repetition):
        print('==========Running Experiment #%d==========' % (i + 1))
        error_hists[i], local_error_hists[i] = IST(X, Y, args.num_neurons, args.num_workers,
                                    args.init_scale, args.probability, 
                                    args.learning_rate, args.local_iter,
                                    args.global_iter, args.mask_type,
                                    args.verbose)
    
    # print(error_hists.mean(axis=0), local_error_hists.mean(axis=0))
    
    info_dict['error_hist_mean'] = error_hists.mean(axis=0).tolist()
    info_dict['error_hist_std'] = error_hists.std(axis=0).tolist()
    info_dict['error_local_mean'] = local_error_hists.mean(axis=0).tolist()
    info_dict['error_local_std'] = local_error_hists.std(axis=0).tolist()
    result_file = open(args.result_fname, 'w+')
    json.dump(info_dict, result_file)
    result_file.close()
