import argparse
import json
import numpy as np
import tensorflow as tf

from resnet_cifar_experiments import IST_Converge, load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=100)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-p', '--num_workers', type=int, default=1)
    parser.add_argument('-w', '--width', type=int, default=0)
    parser.add_argument('-x', '--probability', type=float, default=1.)
    parser.add_argument('-e', '--local_iter', type=int, default=1)
    parser.add_argument('-g', '--global_iter', type=int, default=100)
    parser.add_argument('-k', '--init_scale', type=float, default=1.)
    parser.add_argument('-t', '--mask_type', type=str, choices=['Bernoulli', 'Categorical'], default='Bernoulli')
    parser.add_argument('-r', '--repetition', type=int, default=1)
    parser.add_argument('-f', '--result_fname', type=str, default='result.txt')
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args = parser.parse_args()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    X, Y = load_data(args.num_samples)
    hidden_neurons = [250, 500, 750, 1000, 1250, 1500]
    # hidden_neurons = [500, 100, 1500]
    if args.width != 0:
        hidden_neurons = [args.width]
    if args.mask_type == 'Bernoulli':
        result_matrix = np.zeros((len(hidden_neurons), args.repetition))
        for wid_idx, width in enumerate(hidden_neurons):
            for i in range(args.repetition):
                print(f'==========Running Width {width}, Experiment #{i+1}==========')
                result_matrix[wid_idx, i] = IST_Converge(X, Y, width,
                                                        args.num_workers,
                                                        args.init_scale,
                                                        args.probability,
                                                        args.learning_rate,
                                                        args.local_iter,
                                                        args.mask_type,
                                                        args.verbose)
                print(f'The error is {result_matrix[wid_idx, i]:.5f}')
    
        result_dict = dict()
        result_dict['err_mean'] = np.mean(result_matrix, axis=1).tolist()
        result_dict['err_var'] = np.std(result_matrix, axis=1).tolist()
        with open(args.result_fname, 'w+') as result_file:
            json.dump(result_dict, result_file)
        
    
    
    
    
    
