import nest_asyncio
nest_asyncio.apply()
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import layers
from absl import logging, app, flags
from typing import Callable, Tuple, Optional
import functools
import random
import logging 
import logging.handlers
import attr
from models import cnn
from models import rnn
from models import cifar_cnn
from datasets import emnist
from datasets import shakespeare
from datasets import cifar100
from optimizers.customAdam import build_custom_adam
from optimizers.customAdaDB import build_custom_adadb
from optimizers.customSGDm import build_custom_sgdm
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from tensorflow.keras import mixed_precision
import gc
import pickle
import sys


np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# Mixed precision training. 
tff.backends.native.set_sync_local_cpp_execution_context()
policy = mixed_precision.Policy('mixed_float16')
#policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
  raise ValueError('Cannot detect physical GPU device in TF')
tf.config.list_logical_devices()

########## SETTINGS ############
#General
learning_rate_index=0
acceptable_tuning_acc=40
rounds_per_eval=5
train_clients_per_round=5 # 5 10 50
client_epochs_per_round=1 #'Number of epochs in the client to take per round #1 2 4 8 16
server_learning_rate=1.0
client_learning_rate=0.1
adadb_learning_rate=1.0 #10 1.0 0.1 0.01 0.001
batch_size=16 # 'Batch size used on the client (This higher-level batch size determines how many client updates are aggregated before sending them to the central server)
test_batch_size=128 #cnn128 lstm16/100 #Minibatch size of test data (the number of data samples processed together in a single iteration on a client device)
validation_batch_size=50
num_validation_examples=10000
validation_window=10 #tuning: 100 stack/10 emnist training: 400 stack/100 emnist 
convergence_percentage=45
sma_window_size=5

#CNN
buffer_size=418
drop_remainder=False
only_digits=False

#CNN2
cifar_buffer_size=10000

#RNN
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# The unique characters in the file
vocab = sorted(set(text))
# Length of the vocabulary in StringLookup Layer
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
vocab_size_rnn = len(ids_from_chars.get_vocabulary())
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
seq_length = 100
ss_batch_size=8
ss_buffer_size=100
ss_drop_remainder=True

#fedAdam and fedYogi
#Fixed beta based on https://arxiv.org/abs/2003.00295
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-3
adadb_epsilon = 1e-4

######### FUNCTION DEFINITION ##########

def get_avg_metrics(csv_file, column_name, metric, window_size, convergence_percentage=None):
    data = pd.read_csv(csv_file)

    # Extract data from the specified column
    data_column = data[column_name]
    if metric == "convergence":
        # Calculate the moving average line
        moving_average = data_column.rolling(window=window_size).mean()
    
        # Find the round where the moving average crosses target convergence percentage
        metrics = next((i for i, avg in enumerate(moving_average) if avg > convergence_percentage), None)
    elif metric == "avgAcc":
        # Calculate the average accuracy over window_size
        metrics = data_column.tail(window_size).mean()
    return metrics

    
def plot_and_save_csv(csv_file, column_name, window_size):
    # Load data from the CSV file
    data = pd.read_csv(csv_file)
    output_filename = os.path.splitext(os.path.basename(csv_file))[0] + "_" + column_name + ".jpg"   
    # Extract data from the specified column
    data_column = data[column_name]
    if len(data_column) < total_rounds:
        acc_points=int(total_rounds/rounds_per_eval)
        data_series = pd.Series(data_column)
    else:
        acc_points=len(data_column)
        data_series=data_column
    x_values = [i * rounds_per_eval for i in range(acc_points)]
    x_series = pd.Series(x_values[:acc_points])
    # Calculate the moving average line
    moving_average = data_series.rolling(window=window_size).mean()
    # Plot the data and the moving average line
    plt.clf()
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(x_series, data_column, label=column_name)
    plt.plot(x_series, moving_average, label=f'{window_size}-Round Moving Average', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel(column_name)
    plt.title(f'{column_name} over Rounds')
    plt.legend(loc='lower right')

    # Save the plot to the specified output file
    plt.savefig(results_path + "/" + output_filename)

    # Display the plot
    plt.show()

def plot_heatmap_from_csv(csv_file, server_learning_rates, client_learning_rates):
    # Load the results matrix from the CSV file
    results_matrix = np.genfromtxt(csv_file, delimiter=',')
    output_filename = os.path.splitext(os.path.basename(csv_file))[0] + ".jpg"
    formatted_server_rates = ["{:.3f}".format(rate) for rate in server_learning_rates]
    formatted_client_rates = ["{:.3f}".format(rate) for rate in client_learning_rates]
    plt.clf()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Set axis labels and title
    ax.set_xlabel('Client Learning Rates',labelpad=4)
    ax.set_ylabel('Server Learning Rates',labelpad=4)
    ax.set_title('Validation Accuracy Grid')

    # Define a colormap for shading based on accuracy
    cmap = plt.get_cmap('YlGn')  # Adjust the colormap to your preference
    
    # Create a heatmap
    heatmap = ax.pcolormesh(np.arange(len(server_learning_rates)), np.arange(len(client_learning_rates)), results_matrix, cmap=cmap, shading='auto')
    
    # Set the tick locations and labels with formatted rates
    ax.set_xticks(np.arange(len(server_learning_rates)), minor=False)
    ax.set_xticklabels(formatted_server_rates, rotation=45)
    ax.set_yticks(np.arange(len(client_learning_rates)), minor=False)
    ax.set_yticklabels(formatted_client_rates)
    
    # Display the accuracy values in the cells with 2 decimal point precision
    for i in range(len(server_learning_rates)):
        for j in range(len(client_learning_rates)):
            text = ax.text(
                i,  # Center of the cell in the x-direction
                j,  # Center of the cell in the y-direction
                f'{results_matrix[j, i]:.2f}',  # Use results_matrix[j][i] to match the data order
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                color='black'
            )
    # Add a color bar on the side to represent the accuracy range
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Accuracy')
    
    # Save the plot to the specified output file
    plt.savefig(results_path + "/" + output_filename)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="This is a main script to train federated learning models and tune hyperparameters")

    # Define command line arguments
    parser.add_argument('-a', '--algorithm', choices=['fedAvg', 'fedAdam', 'fedAdadb',], required=True, help="Specify the algorithm (fedAvg, fedAdam, or fedAdadb)")
    parser.add_argument('-d', '--dataset', choices=['emnist', 'shakespeare', 'cifar100'], required=True, help="Specify the dataset (emnist, shakespeare , or 'cifar100')")
    parser.add_argument('-t', '--task', choices=['training', 'tuning'], required=True, help="Specify the task (training or tuning)")
    parser.add_argument('-r', '--rounds', type=int, required=True, help="Specify the number of rounds")
    parser.add_argument('-s', '--store', choices=['nightly', 'official'], required=True, help="Store results as nightly or official")
    parser.add_argument('-g', '--gpu', type=int, required=False, help="select a gpu 0,1,2")
    parser.add_argument('-e', '--epoch', type=int, required=False, help="select an epoch size 1 2 4 8 16")
    parser.add_argument('-c', '--cohort', type=int, required=False, help="select a cohort size 5 50 500")
    
    args = parser.parse_args()
    global algorithm, dataset, task, total_rounds, tunning_rounds, federated_train_data, federated_validation_data, results_path, store, validation_window
    # Access the selected options
    algorithm = args.algorithm
    dataset = args.dataset
    task = args.task
    total_rounds = args.rounds
    tuning_rounds = args.rounds
    store = args.store
    gpu = args.gpu
    train_clients_per_round = args.cohort
    client_epochs_per_round = args.epoch

    #devices = ["/gpu:"+str(gpu)]  # Specify the GPU devices to use
    #strategy = tf.distribute.MirroredStrategy(devices=devices)

    print(f"##### SETTINGS #####")
    print(f"Algorithm: {algorithm}")
    print(f"Dataset: {dataset}")
    print(f"Task: {task}")
    print(f"Store as: {store}")
    print(f"####################")
    
    #############################
    # Create the results dir
    #############################
    results_dir = 'results/' + store
    results_path = results_dir + "/" + dataset + "/" + algorithm + "/" + task
    # Check if the directory already exists
    if not os.path.exists(results_path):
        # If it doesn't exist, create the directory
        os.makedirs(results_path)
        print(f"Directory '{results_path} created.")
    else:
        print(f"Directory '{results_path}' already exists")
        
    ##################
    # Load datasets
    ##################
    if dataset == "emnist":
        federated_train_data, _ = emnist.get_federated_emnist_datasets(
            batch_size=batch_size, # 'Batch size used on the client
            test_batch_size=test_batch_size, #Minibatch size of test data
            train_clients_per_round=train_clients_per_round,
            client_epochs_per_round=client_epochs_per_round,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
        )
        _, federated_validation_data, federated_test_data =  emnist.get_centralized_emnist_datasets(
            batch_size=batch_size, # 'Batch size used on the client
            test_batch_size=test_batch_size, #Minibatch size of test data
            train_clients_per_round=train_clients_per_round,
            client_epochs_per_round=client_epochs_per_round,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
        )
    elif dataset == "shakespeare":
        federated_train_data = shakespeare.get_federated_shakespeare_datasets(
            batch_size=ss_batch_size, # 'Batch size used on the client
            test_batch_size=ss_batch_size, #Minibatch size of test data
            client_epochs_per_round=client_epochs_per_round,
            buffer_size=ss_buffer_size,
            drop_remainder=ss_drop_remainder,
            seq_length=seq_length
        )
        _, federated_validation_data, federated_test_data =  shakespeare.get_centralized_shakespeare_datasets(
            batch_size=ss_batch_size, # 'Batch size used on the client
            test_batch_size=ss_batch_size, #Minibatch size of test data
            client_epochs_per_round=client_epochs_per_round,
            buffer_size=ss_buffer_size,
            drop_remainder=ss_drop_remainder,
            seq_length=seq_length
        )
    elif dataset == "cifar100":
        federated_train_data, _ = cifar100.get_federated_cifar100_datasets(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_clients_per_round=train_clients_per_round,
            client_epochs_per_round=client_epochs_per_round,
            buffer_size=cifar_buffer_size,
            drop_remainder=drop_remainder,
        )
        _, federated_validation_data, federated_test_data = cifar100.get_centralized_cifar100_datasets(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_clients_per_round=train_clients_per_round,
            client_epochs_per_round=client_epochs_per_round,
            buffer_size=cifar_buffer_size,
            drop_remainder=drop_remainder,
        )
    else:
        print("invalid dataset option. Please select emnist or stackoverflow")
    
    input_spec=federated_train_data.element_type_structure
    print(f"DEBUG: Loaded datasets")

    #####################################
    # MODEL FN
    #####################################
    def model_fn():
        input_spec=federated_train_data.element_type_structure
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        """Constructs a fully initialized model for use in federated averaging."""
        if dataset == "emnist":
            keras_model = cnn.create_cnn_model(only_digits=only_digits)
        elif dataset == "shakespeare":
            keras_model = rnn.create_rnn_model(vocab_size_rnn, embedding_dim, rnn_units)
            metrics = [rnn.FlattenedCategoricalAccuracy()]
        elif dataset == "cifar100":
            keras_model = cifar_cnn.create_cifar100_cnn_model()
        else:
            print("invalid dataset option. Please select emnist cifar100 or shakespeare")
            
        return tff.learning.models.from_keras_model(
            keras_model,
            loss=loss,
            metrics=metrics,
            input_spec=input_spec,
        )
    if task == "training":
        if dataset == "emnist":
            validation_window=100
        if dataset == "shakespeare":
            validation_window=100
        if dataset == "cifar100":
            validation_window=100
        #######################################################################################
        # Getting the best learning rates if there has been a tuning before
        #######################################################################################
        server_learning_rates = np.logspace(-3, 1, num=9)
        client_learning_rates = np.logspace(-3, 1, num=9)
        tuning_path = results_dir + "/" + dataset + "/" + algorithm + "/tuning"
        tuning_file_specific = tuning_path + "/c" + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + "_tuning_data.csv"
        tuning_file_default = tuning_path + "/tuning_data.csv"
        tuning_file_default2 = tuning_path + "/c5_e1_tuning_data.csv"
        checkpoint_file = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_checkpoint.pkl'
        intermediate_round_file = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_intermediate_round.txt'
        pickle_flag=False
        if os.path.exists(tuning_file_specific):
            tuning_file = tuning_file_specific
        elif os.path.exists(tuning_file_default):
            tuning_file = tuning_file_default
        elif os.path.exists(tuning_file_default2):
            tuning_file = tuning_file_default2
        else:
            tuning_file="NAN"
        
        if os.path.exists(tuning_file):
            data = pd.read_csv(tuning_file, header=None)
            accuracy_array = data.values
            # Find indices where accuracy is greater than 90%
            i_indices, j_indices = np.where(accuracy_array > acceptable_tuning_acc)            
            # Get accuracy values corresponding to the indices
            accuracies = accuracy_array[(i_indices, j_indices)]
            # Create a list of tuples with i, j pairs and their corresponding accuracy
            ij_accuracy_pairs = list(zip(i_indices, j_indices, accuracies))
            # Sort the list based on accuracy values in descending order
            sorted_ij_pairs = sorted(ij_accuracy_pairs, key=lambda x: x[2], reverse=True)
            # Extract sorted i, j pairs where accuracy is more than 90%
            sorted_ij_pairs_above_90_percent = [(i, j) for i, j, _ in sorted_ij_pairs]
            print(sorted_ij_pairs_above_90_percent[learning_rate_index])
            #max_index = data.stack().idxmax()
            #i_index, j_index = max_index      
            i_index, j_index = sorted_ij_pairs_above_90_percent[learning_rate_index]  
            server_learning_rate = float(server_learning_rates[i_index])
            client_learning_rate = float(client_learning_rates[j_index])
        else:
            server_learning_rate=1.0
            client_learning_rate=0.1
        print(server_learning_rate)
        print(client_learning_rate)
        ######################################
        #   INITIALIZE
        ######################################
        #with strategy.scope():
        def server_optimizer_fn():
            return tf.keras.optimizers.SGD(learning_rate=server_learning_rate, clipnorm=1e10)
       
        def client_optimizer_fn():
            return tf.keras.optimizers.SGD(learning_rate=client_learning_rate)
       
        fed_adam_server_optimizer=build_custom_adam(
            learning_rate= server_learning_rate,
            beta_1 = beta_1,
            beta_2 = beta_2,
            epsilon = epsilon,
            clipnorm=1e10
        )
        fed_adadb_server_optimizer=build_custom_adadb(
            alpha_star = adadb_learning_rate,
            learning_rate= server_learning_rate,
            beta_1 = beta_1,
            beta_2 = beta_2,
            epsilon = adadb_epsilon,
            clipnorm = 1e10
        )
        if algorithm == "fedAvg":
            iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                    model_fn=model_fn,
                    client_optimizer_fn=client_optimizer_fn,
                    server_optimizer_fn=server_optimizer_fn,
                    use_experimental_simulation_loop=True)
        elif algorithm == "fedAdam":
            iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                    model_fn=model_fn,
                    client_optimizer_fn=client_optimizer_fn,
                    server_optimizer_fn=fed_adam_server_optimizer,
                    use_experimental_simulation_loop=True)
        elif algorithm == "fedAdadb":
            iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                    model_fn=model_fn,
                    client_optimizer_fn=client_optimizer_fn,
                    server_optimizer_fn=fed_adadb_server_optimizer,
                    use_experimental_simulation_loop=True)
        else:
            raise ValueError("Unsupported algorithm: " + algorithm)
        try:
            with open(results_path + "/" + checkpoint_file, 'rb') as file:
                server_state = pickle.load(file)
                pickle_flag=True
        except FileNotFoundError:
            server_state = iterative_process.initialize()
            print(f"DEBUG: Initialized model")
        try:
            with open(results_path + "/" + intermediate_round_file, 'r') as rfile:
                starting_round = int(rfile.read().strip())
        except FileNotFoundError:
            starting_round = 0    

        if dataset == "emnist":
            keras_model = cnn.create_cnn_model(only_digits=only_digits)
        elif dataset == "shakespeare":
            keras_model = rnn.create_rnn_model(vocab_size_rnn, embedding_dim, rnn_units)
        elif dataset == "cifar100":
            keras_model = cifar_cnn.create_cifar100_cnn_model()
        else:
            print("invalid dataset option")
        model_summary = keras_model.summary()
        print(model_summary)     
        ###################################
        #    TRAINING LOOP
        ###################################
        train_loss=[]
        train_acc=[]
        validation_acc=[]
        convergence_rounds="did not converge"
        # Specify the CSV file path
        training_csv = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_training_data.csv'
        validation_csv = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_validation_data.csv'
        training_settings = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_training_settings.txt'
        model_path = results_path + "/model"
        total_clients = len(federated_train_data.client_ids)
        # Initialize variables to keep track of time
        total_start_time = time.time()

        if pickle_flag == False:
            # If the file doesn't exist, initialize the results csvs
            with open(results_path + "/" + validation_csv, mode='w', newline='') as file_val:
                writer = csv.writer(file_val)
                # Write the header row
                writer.writerow(['Round', 'Accuracy'])
            file_val.close()
            
            with open(results_path + "/" + training_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header row
                writer.writerow(['Round', 'Accuracy', 'Loss', 'Clients_num', 'Examples', 'Batches', 'TimePerRound'])
            file.close()
        print(f"DEBUG: Starting training loop...")
        for round_num in range(starting_round, total_rounds):
            # Start the timer for the current round
            round_start_time = time.time()
            sampled_clients = np.random.choice(
                federated_train_data.client_ids, size=train_clients_per_round, replace=True
            )
            sampled_train_data = [
                federated_train_data.create_tf_dataset_for_client(client)
                for client in sampled_clients
            ]
            #with strategy.scope():
            server_state, train_metrics = iterative_process.next(
                server_state, sampled_train_data
            )
            print(f'Round {round_num}')
            print(f'\tTraining metrics: {train_metrics}')
            gc.collect()
            #tf.keras.backend.clear_session()
            #loss, accuracy = evaluate_model(keras_model, federated_test_data)
            train_loss.append(train_metrics['client_work']['train']['loss'])
            train_acc.append(train_metrics['client_work']['train']['sparse_categorical_accuracy'])
            # Calculate the time taken for this round
            round_end_time = time.time()
            round_elapsed_time = round_end_time - round_start_time

            # Open the CSV file in write mode
            with open(results_path + "/" + training_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write the data rows
                writer.writerow([round_num,
                              train_metrics['client_work']['train']['sparse_categorical_accuracy'], 
                              train_metrics['client_work']['train']['loss'],
                              train_clients_per_round,
                              train_metrics['client_work']['train']['num_examples'],
                              train_metrics['client_work']['train']['num_batches'],
                              round_elapsed_time])
            #sampled_train_data = None
            gc.collect()
            #tf.keras.backend.clear_session()
            if round_num % rounds_per_eval == 0:
                #server_state.model.assign_weights_to(keras_model)
                server_state.global_model_weights.assign_weights_to(keras_model)
                if dataset == "emnist":
                    accuracy = cnn.evaluate_cnn_model(keras_model, federated_validation_data)
                    if isinstance(accuracy, tf.Tensor) and accuracy.dtype == tf.float32:
                        accuracy = float(accuracy.numpy())
                elif dataset == "shakespeare":
                    loss, accuracy = rnn.evaluate_rnn_model(keras_model, federated_validation_data)
                elif dataset == "cifar100":
                    accuracy = cifar_cnn.evaluate_cifar100_cnn_model(keras_model, federated_validation_data)
                    if isinstance(accuracy, tf.Tensor) and accuracy.dtype == tf.float32:
                        accuracy = float(accuracy.numpy())
                else:
                    print("invalid dataset option. Please select emnist or stackoverflow")
                validation_acc.append(accuracy * 100)
                print(f'\tValidation accuracy: {accuracy * 100.0:.2f}%')
                with open(results_path + "/" + validation_csv, mode='a', newline='') as file_val:
                    writer = csv.writer(file_val)
                    # Write the data rows
                    writer.writerow([round_num, accuracy * 100])
                if dataset == "stackoverflow":
                    loss, accuracy = None, None
                    gc.collect()
                    #tf.keras.backend.clear_session()
            #save training state at a checkpoint every 100 rounds
            if round_num % 100 == 0:
                # Save intermediate results
                with open(results_path + "/" + checkpoint_file, 'wb') as file:
                    pickle.dump(server_state, file) 
                with open(results_path + "/" + intermediate_round_file, 'w') as rfile:
                    rfile.write(str(round_num))
                

        # Calculate the total time elapsed for training
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        convergence_rounds=get_avg_metrics(results_path + "/" + validation_csv, "Accuracy", "convergence", sma_window_size, convergence_percentage)
        final_average_acc = get_avg_metrics(results_path + "/" + validation_csv, "Accuracy", "avgAcc", validation_window)
        max_convergence_rounds=get_avg_metrics(results_path + "/" + validation_csv, "Accuracy", "convergence", sma_window_size, 60)
                          
        # Open a file for writing (you can specify the file path)
        with open(results_path + "/" + training_settings, 'w') as file_settings:
            # Write the variables to the file
            file_settings.write(f"total_clients: {total_clients}\n")
            file_settings.write(f"client_epochs_per_round: {client_epochs_per_round}\n")
            file_settings.write(f"clients_per_round: {train_clients_per_round}\n")
            file_settings.write(f"server_learning_rate: {server_learning_rate}\n")
            file_settings.write(f"client_learning_rate: {client_learning_rate}\n")
            file_settings.write(f"total_time_elapsed: {total_elapsed_time}\n")
            file_settings.write(f"convergence_rounds: {convergence_rounds}\n")
            file_settings.write(f"max_convergence_rounds: {max_convergence_rounds}\n")
            file_settings.write(f"Accuracy over last {validation_window} rounds: {final_average_acc}\n")
        # Save the model wights
        tf.keras.models.save_model(keras_model, model_path)
        
        file_val.close()
        file.close()
        #file_settings.close()
        del keras_model
        gc.collect()
        tf.keras.backend.clear_session()
        
        plot_and_save_csv(results_path + "/" +  training_csv, 'Accuracy', sma_window_size)
        plot_and_save_csv(results_path + "/" + training_csv, 'Loss', sma_window_size)
        plot_and_save_csv(results_path + "/" + validation_csv, 'Accuracy', sma_window_size)
        return True
####################################
# TUNING
# ###################################
    elif task == "tuning":
        if dataset == "emnist":
            validation_window=10
        if dataset == "shakespeare":
            validation_window=10
        if dataset == "cifar100":
            validation_window=10
        #with strategy.scope():
        tuning_csv = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_tuning_data.csv'
        checkpoint_file = 'c' + str(train_clients_per_round) + '_e' + str(client_epochs_per_round) + '_checkpoint.pkl'
        server_learning_rates = np.logspace(-3, 1, num=9)
        client_learning_rates = np.logspace(-3, 1, num=9)
        if algorithm == "fedSgd":
            client_learning_rates=[1]
        
        try:
            with open(results_path + "/" + checkpoint_file, 'rb') as file:
                results_matrix = pickle.load(file)
                has_non_zero = np.any(results_matrix != 0)
                if has_non_zero:
                    found_zero = False
                    first_zero_index = None
                    
                    for i in range(results_matrix.shape[0]):
                        for j in range(results_matrix.shape[1]):
                            if results_matrix[i, j] == 0:
                                first_zero_index = (i, j)
                                found_zero = True
                                break
                        if found_zero:
                            break
                    if found_zero:
                        print(f"The indices of the first zero value: {first_zero_index}")
        except FileNotFoundError:
            # If the file doesn't exist, initialize the results matrix
            results_matrix = np.zeros((len(server_learning_rates), len(client_learning_rates)))
            first_zero_index = [0, 0]

        i_start_from=first_zero_index[0]
        j_start_from=first_zero_index[1]
        for i in range(i_start_from, len(server_learning_rates)):
            for j in range(j_start_from, len(client_learning_rates)):
                server_learning_rate=float(server_learning_rates[i])
                client_learning_rate=float(client_learning_rates[j])
                def server_optimizer_fn():
                    return tf.keras.optimizers.SGD(learning_rate=server_learning_rate, clipnorm=1e10)
                def client_optimizer_fn():
                    return tf.keras.optimizers.SGD(learning_rate=client_learning_rate)
                fed_adam_server_optimizer=tff.learning.optimizers.build_adam(
                    learning_rate= server_learning_rate,
                    beta_1 = beta_1,
                    beta_2 = beta_2,
                    epsilon = epsilon
                )
                fed_adadb_server_optimizer=build_custom_adadb(
                    alpha_star = adadb_learning_rate,
                    learning_rate= server_learning_rate,
                    beta_1 = beta_1,
                    beta_2 = beta_2,
                    epsilon = adadb_epsilon,
                    clipnorm = 1e10
                )
                #re initialize the model before tunning with new rates
                if algorithm == "fedAvg":
                    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                            model_fn=model_fn,
                            client_optimizer_fn=client_optimizer_fn,
                            server_optimizer_fn=server_optimizer_fn,
                            use_experimental_simulation_loop=True)
                elif algorithm == "fedAdam":
                    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                            model_fn=model_fn,
                            client_optimizer_fn=client_optimizer_fn,
                            server_optimizer_fn=fed_adam_server_optimizer,
                            use_experimental_simulation_loop=True)
                elif algorithm == "fedAdadb":
                    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                            model_fn=model_fn,
                            client_optimizer_fn=client_optimizer_fn,
                            server_optimizer_fn=fed_adadb_server_optimizer,
                            use_experimental_simulation_loop=True)
                else:
                    raise ValueError("Unsupported algorithm: " + algorithm)
                server_state = iterative_process.initialize()
                if dataset == "emnist":
                    keras_model = cnn.create_cnn_model(only_digits=only_digits)
                elif dataset == "shakespeare":
                    keras_model = rnn.create_rnn_model(vocab_size_rnn, embedding_dim, rnn_units)
                elif dataset == "cifar100":
                    keras_model = cifar_cnn.create_cifar100_cnn_model()
                else:
                    print("invalid dataset option. Please select emnist or stackoverflow")
                total_clients = len(federated_train_data.client_ids)
                # Initialize variables to keep track of time
                total_start_time = time.time()
                
                validation_acc = []
                for round_num in range(tuning_rounds):
                    # Start the timer for the current round
                    sampled_clients = np.random.choice(
                        federated_train_data.client_ids, size=train_clients_per_round, replace=False
                    )
                    sampled_train_data = [
                        federated_train_data.create_tf_dataset_for_client(client)
                        for client in sampled_clients
                    ]
                    #with strategy.scope():
                    server_state, train_metrics = iterative_process.next(
                        server_state, sampled_train_data
                        )
                    print(f'Round {round_num}')
                    print(f'\tTraining metrics: {train_metrics}')
                    gc.collect()
                    tf.keras.backend.clear_session()
                    if round_num >= tuning_rounds - validation_window:
                        #server_state.model.assign_weights_to(keras_model)
                        server_state.global_model_weights.assign_weights_to(keras_model)
                        if dataset == "emnist":
                            accuracy = cnn.evaluate_cnn_model(keras_model, federated_validation_data)
                        elif dataset == "shakespeare":
                            loss, accuracy = rnn.evaluate_rnn_model(keras_model, federated_validation_data)
                        elif dataset == "cifar100":
                            accuracy = cifar_cnn.evaluate_cifar100_cnn_model(keras_model, federated_validation_data)
                        else:
                            print("invalid dataset option. Please select emnist or stackoverflow")
                        validation_acc.append(accuracy * 100.0)
                        print(f'\tValidation accuracy {accuracy * 100.0:.2f}%')
                mean_val_acc = sum(validation_acc) / len(validation_acc)
                print(f'\tValidation accuracy over last {validation_window} rounds: {mean_val_acc:.2f}%')
                results_matrix[i][j] = mean_val_acc
                print(f'client learning rate: {client_learning_rate}')
                print(f'server learning rate: {server_learning_rate}')
                        
                combined_progress = (i * len(client_learning_rates)) + j
                print(f"Progress: {combined_progress} out of {len(server_learning_rates) * len(client_learning_rates)}\n")
                # Save intermediate results
                with open(results_path + "/" + checkpoint_file, 'wb') as file:
                    pickle.dump(results_matrix, file)       
                                                     
                # Calculate the total time elapsed for training
                total_end_time = time.time()
                total_elapsed_time = total_end_time - total_start_time
                del keras_model
                gc.collect()
                tf.keras.backend.clear_session()
                print(f'\tTraining time: {total_elapsed_time}')
            j_start_from=0
        # Save the results matrix to a CSV file
        np.savetxt(results_path + "/" + tuning_csv, results_matrix, delimiter=',')
        # Plot heatmap
        plot_heatmap_from_csv(results_path + "/" + tuning_csv, server_learning_rates, client_learning_rates)
         # Open a file for writing (you can specify the file path)
        with open(results_path + "/c" + str(train_clients_per_round) + "_e" + str(client_epochs_per_round) + "_tuning_settins.txt", 'w') as file_settings:
            # Write the variables to the file
            file_settings.write(f"total_clients: {total_clients}\n")
            file_settings.write(f"client_epochs_per_round: {client_epochs_per_round}\n")
            file_settings.write(f"clients_per_round: {train_clients_per_round}\n")
            file_settings.write(f"Accuracy Validation Window (in rounds): {validation_window}\n")
            file_settings.write(f"Training rounds: {total_rounds}\n")
            file_settings.write(f"total_time_elapsed: {total_elapsed_time}\n")
        
        print("Finished!")
    return True


if __name__ == "__main__":
    main()
