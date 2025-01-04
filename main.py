from __future__ import print_function

import aggregation_rules
import numpy as np
import random
import argparse
import attacks
import data_loaders_esol

import os
import math
import subprocess

import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from models.gcn import GCN
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from torch_geometric.nn import GCNConv
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import datetime
import shutil
import time


def parse_args():
    """
    Parses all commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description="SAFEFL: MPC-friendly framework for Private and Robust Federated Learning")

    ### Model and Dataset
    parser.add_argument("--net", help="net: GCN | lr", type=str, default="GCN")
    parser.add_argument("--task_type", help="classification | regression", type=str, default="regression")
    parser.add_argument("--dataset", help="dataset: HAR | ESOL", type=str, default="ESOL")
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-38)
    parser.add_argument("--adam_lr", help="adam learning rate for the local trainings", type=float, default=0.0007)
    parser.add_argument("--n_client", help="# parties", type=int, default=3)
    parser.add_argument('--use_agreggation', default=True, help='use Adam optimizer or FL aggregations')

    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--p", help="bias probability of class 1 in server dataset", type=float, default=0.1)

    ### Training
    parser.add_argument("--global_epoch", help="# iterations", type=int, default=10)
    parser.add_argument("--local_epoch", help="# local optimization or train", type=int, default=500)
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--gpu", help="no gpu = -1, gpu training otherwise", type=int, default=-1)
    parser.add_argument("--seed", help="seed", type=int, default=1)
    parser.add_argument("--nruns", help="number of runs for averaging accuracy", type=int, default=1)
    parser.add_argument("--test_every", help="testing interval", type=int, default=1)

    ### Aggregations
    parser.add_argument("--aggregation", help="fedavg or fltrust or", type=str, default="fltrust")

    # FLOD
    parser.add_argument("--flod_threshold", help="hamming distance threshold as fraction of total model parameters",
                        type=float, default=0.5)

    # FLAME
    parser.add_argument("--flame_epsilon", help="epsilon for differential privacy in FLAME", type=int, default=3000)
    parser.add_argument("--flame_delta", help="delta for differential privacy in FLAME", type=float, default=0.001)

    # DNC
    parser.add_argument("--dnc_niters", help="number of iterations to compute good sets in DnC", type=int, default=5)
    parser.add_argument("--dnc_c", help="filtering fraction, percentage of number of malicious clients filtered",
                        type=float, default=1)
    parser.add_argument("--dnc_b", help="dimension of subsamples must be smaller, then the dimension of the gradients",
                        type=int, default=2000)

    ### Attacks
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=1)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="min_sum_attack",
                        choices=["no", "trim_attack", "krum_attack",
                                 "scaling_attack", "fltrust_attack", "label_flipping_attack", "min_max_attack",
                                 "min_sum_attack"])

    ### MP-SPDZ
    parser.add_argument('--qsmc', default=True, action='store_true', help='Run example in multiprocess mode')
    parser.add_argument("--port", help="port for the qsmc servers", type=int, default=38053)
    parser.add_argument("--chunk_size", help="data amount send between client and server at once", type=int,
                        default=200)
    parser.add_argument("--protocol", help="protocol used in QSMC", type=str, default="mascot",
                        choices=["semi2k", "mascot", "spdz2k", "replicated2k", "psReplicated2k"])
    parser.add_argument("--players", help="number of computation parties during aggregation", type=int, default=2)
    parser.add_argument("--threads", help="number of threads per computation party in MP-SPDZ", type=int, default=1)
    parser.add_argument("--parallels", help="number of parallel computation for each thread", type=int, default=1)
    parser.add_argument('--always_compile', default=True, action='store_true',
                        help='compiles program even if it was already compiled')

    return parser.parse_args()


def display_predict(pred1, target1, pred2, target2):
    df = pd.DataFrame()
    df["y_real"] = target1
    df["y_pred"] = pred1

    df["y_real"] = df["y_real"].apply(lambda row: row[0])
    df["y_pred"] = df["y_pred"].apply(lambda row: row[0])

    # Determine the limits for the plot based on real values
    min_val = df["y_real"].min()
    max_val = df["y_real"].max()

    axes = sns.scatterplot(data=df, x="y_real", y="y_pred", color='red')
    axes.set_xlabel("Real Solubility")
    axes.set_ylabel("Predicted Solubility")

    # Set the same scale for both axes based on real values
    axes.set_xlim(min_val, max_val)
    axes.set_ylim(min_val, max_val)

    plt.show()


def predict(model, test_loader, device, epoch, display=False, save=None):
    all_preds = []
    all_targets = []

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            inputs_x, inputs_edge_index, inputs_batch, targets = batch.x, batch.edge_index, batch.batch, batch.y
            inputs_x = inputs_x.to(device)
            inputs_edge_index = inputs_edge_index.to(device)
            inputs_batch = inputs_batch.to(device)
            targets = targets.to(device)

            preds, logits = model(inputs_x.float(), inputs_edge_index.int(), inputs_batch)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    df = pd.DataFrame()
    df["y_real"] = all_targets
    df["y_pred"] = all_preds

    df["y_real"] = df["y_real"].apply(lambda row: row[0])
    df["y_pred"] = df["y_pred"].apply(lambda row: row[0])

    # Determine the limits for the plot based on real values
    min_val = df["y_real"].min()
    max_val = df["y_real"].max()

    plt.figure()
    axes = sns.scatterplot(data=df, x="y_real", y="y_pred")
    axes.set_xlabel("Real Solubility")
    axes.set_ylabel("Predicted Solubility")

    # Set the same scale for both axes based on real values
    axes.set_xlim(min_val, max_val)
    axes.set_ylim(min_val, max_val)
    if display:
        plt.show()
    if save is not None:
        os.makedirs(save, exist_ok=True)
        plt.savefig(f"{save}/solubility_plot_epoch_{epoch}.png", dpi=300, facecolor='w', bbox_inches='tight')
    plt.close()

    mse = F.mse_loss(torch.tensor(all_targets), torch.tensor(all_preds), reduction="mean")
    return mse

def old_plot_train_loss(values, name, display= False, save=None):
    # Convert values to float and handle inf values
    losses_float = [float(loss) if np.isfinite(loss) else np.nan for loss in values]
    loss_indices = range(len(losses_float))
    ax = sns.lineplot(x=loss_indices, y=losses_float)
    ax.set(xlabel='Epoch', ylabel=f"{name}")
    if display:
        plt.show()
    if save is not None:
        plt.savefig(f"{save}{name}.png")  # Save as a PNG or any other format
        plt.close()


def plot_train_loss(values, name, display=False, save=None):
    # Convert values to float and handle inf/nan values
    losses_float = [float(loss) if np.isfinite(loss) else 0.0 for loss in values]  # Replace NaNs with 0
    #loss_indices = range(len(losses_float))
    loss_indices = list(range(1, len(losses_float) + 1))
    # Create a bar plot
    plt.figure(figsize=(6, 4))  # Set figure size
    plt.bar(loss_indices, losses_float, width=1.0, color='steelblue', edgecolor='none')  # Bar plot
    plt.xlabel('Epoch', fontsize=20)  # Label for x-axis
    plt.ylabel(name, fontsize=20)  # Label for y-axis
    plt.ylim(0, 1)

    # Remove grid and additional style
    plt.gca().set_facecolor('white')  # Ensure background is white
    plt.gcf().set_facecolor('white')  # Set figure background to white
    plt.xticks(fontsize=18)  # Adjust x-axis font size
    plt.yticks(fontsize=18)  # Adjust y-axis font size

    # Display or save the plot
    if display:
        plt.show()
    if save is not None:
        plt.savefig(f"{save}{name}.png", dpi=300, bbox_inches='tight')  # Save with high quality
        plt.close()
 

def get_device(device):
    """
    Selects the device to run the training process on.
    device: -1 to only use cpu, otherwise cuda if available
    """
    if device == -1:
        ctx = torch.device('cpu')
    else:
        ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(ctx)
    return ctx


def get_net(net_type, num_inputs, num_outputs=10):
    """
    Selects the model architecture.
    net_type: name of the model architecture
    num_inputs: number of inputs of model
    num_outputs: number of outputs/classes
    """

    if net_type == "GCN":
        import models.gcn as gcn
        net = GCN(input_dim=num_inputs, output_dim=num_outputs)
        print(net)
    else:
        raise NotImplementedError
    return net


def get_byz(byz_type):
    """
    Gets the attack type.
    byz_type: name of the attack
    """
    if byz_type == "no":
        return attacks.no_byz
    elif byz_type == 'trim_attack':
        return attacks.trim_attack
    elif byz_type == "krum_attack":
        return attacks.krum_attack
    elif byz_type == "scaling_attack":
        return attacks.scaling_attack_scale
    elif byz_type == "fltrust_attack":
        return attacks.fltrust_attack
    elif byz_type == "label_flipping_attack":
        return attacks.no_byz
    elif byz_type == "min_max_attack":
        return attacks.min_max_attack
    elif byz_type == "min_sum_attack":
        return attacks.min_sum_attack
    else:
        raise NotImplementedError


def get_protocol(protocol, players):
    """
    Returns the shell script name and number of players for the protocol.
    protocol: name of the protocol
    players: number of parties
    """
    if players < 2:
        raise Exception("Number of players must at least be 2")

    if protocol == "semi2k":
        return "semi2k.sh", players

    elif protocol == "mascot":
        return "mascot.sh", players

    elif protocol == 'spdz2k':
        return "spdz2k.sh", players

    elif protocol == "replicated2k":
        if players != 3:
            raise Exception("Number of players must be 3 for replicated2k")
        return "ring.sh", 3

    elif protocol == "psReplicated2k":
        if players != 3:
            raise Exception("Number of players must be 3 for psReplicated2k")
        return "ps-rep-ring.sh", 3

    else:
        raise NotImplementedError


def evaluate_accuracy(data_iterator, net, device, trigger, dataset):
    """
    Evaluate the accuracy and backdoor success rate of the model. Fails if model output is NaN.
    data_iterator: test data iterator
    net: model
    device: device used in training and inference
    trigger: boolean if backdoor success rate should be evaluated
    dataset: name of the dataset used in the backdoor attack
    """
    net.eval()
    if dataset == "ESOL":
        total_mse_error = 0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(data_iterator):
                inputs_x, inputs_edge_index, inputs_batch, targets = batch.x, batch.edge_index, batch.batch, batch.y
                inputs_x = inputs_x.to(device)
                inputs_edge_index = inputs_edge_index.to(device)
                inputs_batch = inputs_batch.to(device)
                targets = targets.to(device)

                preds, logits = net(inputs_x.float(), inputs_edge_index.int(), inputs_batch)

                if not torch.isnan(preds).any():
                    mse_error = F.mse_loss(preds, targets, reduction='sum').item()
                    total_mse_error += mse_error
                    total_samples += 1
                else:
                    print("NaN in output of net")
                    raise ArithmeticError

        avg_mse_error = total_mse_error / total_samples
        return avg_mse_error


def plot_results(runs_test_accuracy, runs_backdoor_success, test_iterations, niter):
    """
    Plots the evaluation results.
    runs_test_accuracy: accuracy of the model in each iteration specified in test_iterations of every run
    runs_backdoor_success: backdoor success of the model in each iteration specified in test_iterations of every run
    test_iterations: list of iterations the model was evaluated in
    niter: number of iteration the model was trained for
    """
    test_acc_std = []
    test_error_list = []
    backdoor_success_std = []
    backdoor_success_list = []

    # insert (0,0) as starting point for plot and calculate mean and standard deviation if multiple runs were performed
    if args.nruns == 1:
        if args.byz_type == "scaling_attack":
            runs_backdoor_success = np.insert(runs_backdoor_success, 0, 0, axis=0)
            backdoor_success_list = runs_backdoor_success
            backdoor_success_std = [0 for i in range(0, len(runs_backdoor_success))]
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=0)
        test_error_list = runs_test_accuracy
        test_acc_std = [0 for i in range(0, len(runs_test_accuracy))]
    else:
        if args.byz_type == "scaling_attack":
            runs_backdoor_success = np.insert(runs_backdoor_success, 0, 0, axis=1)
            backdoor_success_list = np.mean(runs_backdoor_success, axis=0)
            backdoor_success_std = np.std(runs_backdoor_success, axis=0)
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=1)
        test_acc_std = np.std(runs_test_accuracy, axis=0)
        test_error_list = np.mean(runs_test_accuracy, axis=0)

    test_iterations.insert(0, 0)
    # Print accuracy and backdoor success rate in array form to console
    print("Test accuracy of runs:")
    print(repr(runs_test_accuracy))
    if args.byz_type == "scaling_attack":
        print("Backdoor attack success rate of runs:")
        print(repr(runs_backdoor_success))

    # Determine in which iteration in what run the highest accuracy was achieved.
    # Also print overall mean accuracy and backdoor success rate
    max_index = np.unravel_index(runs_test_accuracy.argmax(), runs_test_accuracy.shape)
    if args.nruns == 1:
        print(
            "Run 1 in iteration %02d had the highest accuracy of %0.4f" % (max_index[0] * 50, runs_test_accuracy.max()))
    else:
        print("Run %02d in iteration %02d had the highest accuracy of %0.4f" % (
        max_index[0] + 1, max_index[1] * 50, runs_test_accuracy.max()))
        print("The average final accuracy was: %0.4f with an overall average:" % (test_error_list[-1]))
        print(repr(test_error_list))
        if args.byz_type == "scaling_attack":
            print("The average final backdoor success rate was: %0.4f with an overall average:" % backdoor_success_list[
                -1])
            print(repr(backdoor_success_list))
    # Generate plot with two axis displaying accuracy and backdoor success rate over the iterations
    if args.byz_type == "scaling_attack":
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        accuracy_plot = ax1.plot(test_iterations, test_error_list, color='C0', label='accuracy')
        ax1.fill_between(test_iterations, test_error_list - test_acc_std, test_error_list + test_acc_std, color='C0')
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Backdoor success rate')
        backdoor_plot = ax2.plot(test_iterations, backdoor_success_list, color='C1', label='Backdoor success rate')
        ax2.fill_between(test_iterations, backdoor_success_list - backdoor_success_std,
                         backdoor_success_list + backdoor_success_std, color='C1')
        ax2.set_ylim(0, 1)

        lns = accuracy_plot + backdoor_plot
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=0)
        plt.xlim(0, niter)
        plt.title(
            "Test Accuracy + Backdoor success: " + args.net + ", " + args.dataset + ", " + args.aggregation + ", " + args.byz_type + ", nruns " + str(
                args.nruns))
        plt.grid()
        plt.show()
    # Generate plot with only the accuracy as one axis over the iterations
    else:
        plt.plot(test_iterations, test_error_list, color='C0')
        plt.fill_between(test_iterations, test_error_list - test_acc_std, test_error_list + test_acc_std, color='C0')
        plt.title(
            "Test Accuracy: " + args.net + ", " + args.dataset + ", " + args.aggregation + ", " + args.byz_type + ", nruns " + str(
                args.nruns))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.xlim(0, niter)
        plt.ylim(0, 1)
        plt.grid()
        plt.show()


def weight_init(m):
    """
    Initializes the weights of the layer with random values.
    m: the layer which gets initialized
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=2.24)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        # GCNConv doesn't have a single weight attribute, it has parameters
        for param in m.parameters():
            if param.dim() > 1:  # This checks if the parameter is a weight matrix
                nn.init.xavier_uniform_(param, gain=2.24)
            else:  # This initializes bias
                nn.init.zeros_(param)


def train_net(gcn_net, opt, loss_fcn, tr_data, n_epoch, device):
    gcn_net.to(device)
    all_losses = []
    for local_e in range(n_epoch):
        epoch_loss = 0.0
        for index, batch in enumerate(tr_data):
            batch = batch.to(device)
            opt.zero_grad()  # Clear gradients
            pred, embedding = gcn_net(batch.x.float(), batch.edge_index, batch.batch)
            loss = loss_fcn(pred, batch.y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        # Record average loss for the epoch
        all_losses.append(epoch_loss / len(tr_data))
        if local_e % 100 == 0:
            print(f"local_epoch: {local_e}, average_loss: {all_losses[-1]:.4f}")

    return gcn_net, all_losses

def main():
    """
    The main function that runs the entire training process of the model.
    args: arguments defining hyperparameters
    """
    # log the files
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(name=f"./outputs/{formatted_time}/code", exist_ok=True)
    python_files = [os.path.basename(__file__), "aggregation_rules.py", "data_loaders_esol.py", "utils.py"]
    [shutil.copy(file, f"./outputs/{formatted_time}/code") if os.path.exists(file) else print(f"File {file} does not exist.") for file in python_files]

    args = parse_args()  # parse arguments
    # setup
    device = get_device(args.gpu)
    num_inputs, num_outputs, num_labels = data_loaders_esol.get_shapes(args.dataset)
    byz = get_byz(args.byz_type)

    # Print all arguments
    paraString = ('dataset: p' + str(args.p) + '_' + str(args.dataset) + ", server_pc: " + str(
        args.server_pc) + ", bias: " + str(args.bias)
                  + ", n_client: " + str(args.n_client) + ", net: " + str(args.net) + ", global_epoch: " + str(
                args.global_epoch) + ", lr: " + str(args.lr)
                  + ", batch_size: " + str(args.batch_size) + ", nbyz: " + str(
                args.nbyz) + ", attack: " + str(args.byz_type)
                  + ", aggregation: " + str(args.aggregation) + ", FLOD_threshold: " + str(
                args.flod_threshold)
                  + ", Flame_epsilon: " + str(args.flame_epsilon) + ", Flame_delta: " + str(
                args.flame_delta) + ", Number_runs: " + str(args.nruns)
                  + ", DnC_niters: " + str(args.dnc_niters) + ", DnC_c: " + str(
                args.dnc_c) + ", DnC_b: " + str(args.dnc_b)
                  + ", MP-SPDZ: " + str(args.qsmc) + ", Port: " + str(
                args.port) + ", Chunk_size: " + str(args.chunk_size)
                  + ", Protocol: " + args.protocol + ", Threads: " + str(
                args.threads) + ", Parallels: " + str(args.parallels)
                  + ", Seed: " + str(args.seed) + ", Test Every: " + str(args.test_every))
    print(paraString)

    # saving iterations for averaging
    runs_test_accuracy = []
    runs_backdoor_success = []
    test_iterations = []
    backdoor_success_list = []

    # model
    net_client_1 = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    net_client_1.to(device)
    net_client_2 = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    net_client_2.to(device)
    net_client_3 = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    net_client_3.to(device)
    net_server = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    net_server.to(device)
    #net = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    num_params = torch.cat([xx.reshape((-1, 1)) for xx in net_server.parameters()], dim=0).size()[0]  # used for FLOD to determine threshold
    # loss
    loss_fn_client_1 = torch.nn.MSELoss()
    loss_fn_client_2 = torch.nn.MSELoss()
    loss_fn_client_3 = torch.nn.MSELoss()
    loss_fn_server = torch.nn.MSELoss()
    optimizer_client_1 = torch.optim.Adam(net_client_1.parameters(), lr=args.adam_lr)
    optimizer_client_2 = torch.optim.Adam(net_client_2.parameters(), lr=args.adam_lr)
    optimizer_client_3 = torch.optim.Adam(net_client_3.parameters(), lr=args.adam_lr)
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=args.adam_lr)

    # perform parameter checks
    if args.dnc_b > num_params and args.aggregation == "divide_and_conquer":
        args.dnc_b = num_params  # check for condition in description and fix possible error
        print("b was larger than the dimension of gradients. Set to dimension of gradients for correctness!")

    if args.dnc_c * args.nbyz >= args.n_client and args.aggregation == "divide_and_conquer":
        print("DnC removes all gradients during his computation. Lower c or nbyz, or increase number of devices.")

    if args.server_pc == 0 and (args.aggregation in ["fltrust", "flod", "flare"] or args.byz_type == "fltrust_attack"):
        raise ValueError(
            "Server dataset size cannot be 0 when aggregation is FLTrust, MPC FLTrust, FLOD or attack is fltrust attack")

    if args.dataset == "HAR" and args.n_client != 30:
        raise ValueError("HAR only works for 30 workers!")

    # compile server programm for aggregation in MPC
    if args.qsmc:
        script, players = get_protocol(args.protocol, args.players)
        args.script, args.players = script, players

        if args.aggregation == "fedavg":
            args.filename_server = "mpc_fedavg_server"
            num_gradients = args.n_client
        elif args.aggregation == "fltrust":
            args.filename_server = "mpc_fltrust_server"
            num_gradients = args.n_client + 1
        else:
            raise NotImplementedError

        os.chdir("qsmc")

        args.full_filename = f'{args.filename_server}-{args.port}-{num_params}-{num_gradients}-{args.global_epoch}-{args.chunk_size}-{args.threads}-{args.parallels}'

        if not os.path.exists('./Programs/Bytecode'):
            os.mkdir('./Programs/Bytecode')
        already_compiled = len(
            list(filter(lambda f: f.find(args.full_filename) != -1, os.listdir('./Programs/Bytecode')))) != 0

        if args.always_compile or not already_compiled:
            # compile mpc program, arguments -R 64 -X were chosen so that every protocol works
            os.system('./compile.py -F 64 -X ' + args.filename_server + ' ' + str(args.port) + ' ' + str(
                num_params) + ' ' + str(num_gradients) + ' ' + str(args.global_epoch) + ' ' + str(args.chunk_size) + ' ' + str(
                args.threads) + ' ' + str(args.parallels))

        # setup ssl keys
        os.system('Scripts/setup-ssl.sh ' + str(args.players))
        os.system('Scripts/setup-clients.sh 1')

        os.chdir("..")

    # perform multiple runs
    for run in range(1, args.nruns + 1):
        grad_list = []
        test_error_list = []
        test_iterations = []
        backdoor_success_list = []
        server_process = None

        # fix the seeds for deterministic results
        if args.seed > 0:
            args.seed = args.seed + run - 1
            torch.cuda.manual_seed_all(args.seed)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set aggregation specific variables
        if args.aggregation == "shieldfl":
            previous_global_gradient = 0  # important for ShieldFL, all other aggregation rules don't need it
            previous_gradients = []
        elif args.aggregation == "foolsgold":
            gradient_history = [torch.zeros(size=(num_params, 1)).to(device) for i in
                                range(args.n_client)]  # client gradient history for FoolsGold
        elif args.aggregation == "contra":
            gradient_history = [torch.zeros(size=(num_params, 1)).to(device) for i in
                                range(args.n_client)]  # client gradient history for CONTRA
            reputation = torch.ones(size=(args.n_client,)).to(device)  # reputation scores for CONTRA
            cos_dist = torch.zeros((args.n_client, args.n_client), dtype=torch.double).to(
                device)  # pairwise cosine similarity for CONTRA
        elif args.aggregation == "romoa":
            # don't know why they initialize it like this
            previous_global_gradient = torch.cat(
                [param.clone().detach().flatten() for param in net_server.parameters()]).reshape(-1, 1) + torch.normal(mean=0,
                                                                                                                std=1e-7,
                                                                                                                size=(
                                                                                                                num_params,
                                                                                                                1)).to(
                device)
            sanitization_factor = torch.full(size=(args.n_client, num_params),
                                             fill_value=(1 / args.n_client)).to(device)  # sanitization factors for Romoa

        #train_data, test_data = data_loaders_esol.load_data(args.dataset, args.seed)
        esol_data = MoleculeNet(root=".", name="ESOL")
        if len(esol_data) == 0:
            raise ValueError("ESOL data failed to load or is empty.")

        train_client_1 = DataLoader(esol_data[:int(len(esol_data) * 0.2)], batch_size=args.batch_size,  shuffle=True)
        train_client_2 = DataLoader(esol_data[int(len(esol_data) * 0.2):int(len(esol_data) * 0.4)], batch_size=args.batch_size, shuffle=True)
        train_client_3 = DataLoader(esol_data[int(len(esol_data) * 0.4):int(len(esol_data) * 0.6)], batch_size=args.batch_size, shuffle=True)
        train_clients = [train_client_1, train_client_2, train_client_3]
        for idx, client_data in enumerate(train_clients):
            if len(client_data.dataset) == 0:
                raise ValueError(f"Client {idx + 1} received no data.")

        train_server = DataLoader(esol_data[int(len(esol_data) * 0.6):int(len(esol_data) * 0.8)], batch_size=args.batch_size, shuffle=True)
        test_data = DataLoader(esol_data[int(len(esol_data) * 0.8):], batch_size=args.batch_size, shuffle=False)

        # perform data poisoning attacks
        if args.byz_type == "label_flipping_attack":
            each_party_label = attacks.label_flipping_attack(each_party_label, args.nbyz, num_labels)
        elif args.byz_type == "scaling_attack":
            each_party_data, each_party_label = attacks.scaling_attack_insert_backdoor(each_party_data,
                                                                                       each_party_label,
                                                                                       args.dataset,
                                                                                       args.nbyz, device)

        print("Data done")

        # start FLTrust computation parties
        if args.qsmc:
            os.chdir("qsmc")

            print("Starting Computation Parties")
            # start computation servers using a child process to run in parallel
            server_process = subprocess.Popen(["./run_aggregation.sh", args.script, args.full_filename, str(args.players)])
            #os.system(f"./run_aggregation.sh {args.script} {args.full_filename} {args.players}")

            os.chdir("..")

        # training
        client_1_losses = []
        client_2_losses= []
        client_3_losses = []
        server_losses = []
        all_mse = []
        net_server.apply(weight_init)
        net_client_1.apply(weight_init)
        net_client_2.apply(weight_init)
        net_client_3.apply(weight_init)
        mse_client_1 = predict(net_client_1, test_data, device, "Before_train", display=False, save=f"./outputs/{formatted_time}/client_1/")
        mse_client_2 = predict(net_client_2, test_data, device, "Before_train", display=False, save=f"./outputs/{formatted_time}/client_2/")
        mse_client_3 = predict(net_client_3, test_data, device, "Before_train", display=False, save=f"./outputs/{formatted_time}/client_3/")
        print(f"Test the clients' models using MSE metric before training : \t client_1: {mse_client_1};\t client_2: {mse_client_2};\t client_3:{mse_client_3}")
        for global_epoch in range(args.global_epoch):
            print(f"\nGlobal epoch {global_epoch+1}/{args.global_epoch}")
            net_client_1.train()
            net_client_2.train()
            net_client_3.train()
            net_server.train()

            # perform local training for each party
            print("> Train Client_1 and record its gradients: ")
            net_client_1, loss_client_1 = train_net(net_client_1, optimizer_client_1, loss_fn_client_1, train_client_1, n_epoch=args.local_epoch, device=device)
            client_1_losses.extend(loss_client_1)
            grad_list.append([param.grad.clone().detach() for param in net_client_1.parameters()])
            #grad_list.append([torch.clone(param.grad) for param in net_client_1.parameters()])
            print(">> Train Client_2 and record its gradients: ")
            net_client_2, loss_client_2 = train_net(net_client_2, optimizer_client_2, loss_fn_client_2, train_client_2, n_epoch=args.local_epoch, device=device)
            client_2_losses.extend(loss_client_2)
            grad_list.append([param.grad.clone().detach() for param in net_client_2.parameters()])
            #grad_list.append([torch.clone(param.grad) for param in net_client_2.parameters()])
            print(">>> Train Client_3 and record its gradients: ")
            net_client_3, loss_client_3 = train_net(net_client_3, optimizer_client_3, loss_fn_client_3, train_client_3, n_epoch=args.local_epoch, device=device)
            client_3_losses.extend(loss_client_3)
            grad_list.append([param.grad.clone().detach() for param in net_client_3.parameters()])
            #grad_list.append([torch.clone(param.grad) for param in net_client_3.parameters()])

            # compute server update and append it to the end of the list
            if args.aggregation in ["fltrust", "flod"] or args.byz_type == "fltrust_attack":
                print(">>>> Train Server and record its gradients: ")
                net_server, loss_server = train_net(net_server, optimizer_server, loss_fn_server, train_server, n_epoch=args.local_epoch, device=device)
                server_losses.extend(loss_server)
                #grad_list.append([param.grad.clone().detach() for param in net_server.parameters()])
                grad_list.append([torch.clone(param.grad) for param in net_server.parameters()])
                #print("client_1 gradients: ", grad_list[0])
                #print("client_2 gradients: ", grad_list[1])
                #print("client_3 gradients: ", grad_list[2])
                #print("server gradients: ", grad_list[3])
            # perform the aggregation
            # print("train epoch: {} perform the aggregation using {}".format(e, args.aggregation))
            with (torch.no_grad()):
                if args.use_agreggation:
                    if args.qsmc:
                        print("\nUse 'qsmc_aggregation' to aggregate the gradients in the Server to obtain an updated global model.")
                        updated_net=aggregation_rules.qsmc_aggregation(grad_list, net_server, args.lr, args.nbyz, byz,
                                                             device, param_num=num_params, port=args.port,
                                                             chunk_size=args.chunk_size,
                                                             parties=args.players)
                        aggregation_rules.update_model(updated_net, net_client_1, "client_1", args.lr)
                        aggregation_rules.update_model(updated_net, net_client_2, "client_2", args.lr)
                        aggregation_rules.update_model(updated_net, net_client_3, "client_3", args.lr)
                    elif args.aggregation == "fltrust":
                        print("\nUse 'fltrust' to aggregate the gradients in the Server to obtain an updated global model.")
                        aggregation_rules.fltrust(grad_list, net_server, args.lr, args.nbyz, byz, device)

                    elif args.aggregation == "fedavg":
                        print("\nUse 'fedavg' to aggregate the gradients, without using a Server.")
                        data_sizes = [len(test_data)]*(args.n_client)
                        net_client_1 = aggregation_rules.fedavg(grad_list, net_client_1, args.lr, args.nbyz, byz, device, data_sizes)
                        net_client_2 = aggregation_rules.fedavg(grad_list, net_client_2, args.lr, args.nbyz, byz, device, data_sizes)
                        net_client_3 = aggregation_rules.fedavg(grad_list, net_client_3, args.lr, args.nbyz, byz, device, data_sizes)

                    elif args.aggregation == "krum":
                        aggregation_rules.krum(grad_list, net_server, args.lr, args.nbyz, byz, device)

                    elif args.aggregation == "trim_mean":
                        aggregation_rules.trim_mean(grad_list, net_server, args.lr, args.nbyz, byz, device)

                    elif args.aggregation == "median":
                        aggregation_rules.median(grad_list, net_server, args.lr, args.nbyz, byz, device)

                    elif args.aggregation == "flame":
                        aggregation_rules.flame(grad_list, net_server, args.lr, args.nbyz, byz, device,
                                                epsilon=args.flame_epsilon, delta=args.flame_delta)

                    elif args.aggregation == "shieldfl":
                        previous_global_gradient, previous_gradients = aggregation_rules.shieldfl(grad_list, net_server,
                                                                                                  args.lr,
                                                                                                  args.nbyz,
                                                                                                  byz, device,
                                                                                                  previous_global_gradient,
                                                                                                  global_epoch, previous_gradients)

                    elif args.aggregation == "flod":
                        aggregation_rules.flod(grad_list, net_server, args.lr, args.nbyz, byz, device,
                                               threshold=math.floor(num_params * args.flod_threshold))

                    elif args.aggregation == "divide_and_conquer":
                        aggregation_rules.divide_and_conquer(grad_list, net_server, args.lr, args.nbyz, byz,
                                                             device, niters=args.dnc_niters,
                                                             c=args.dnc_c, b=args.dnc_b)

                    elif args.aggregation == "foolsgold":
                        gradient_history = aggregation_rules.foolsgold(grad_list, net_server, args.lr,
                                                                       args.nbyz, byz, device,
                                                                       gradient_history=gradient_history)

                    elif args.aggregation == "contra":
                        gradient_history, reputation, cos_dist = aggregation_rules.contra(grad_list, net_server,
                                                                                          args.lr,
                                                                                          args.nbyz, byz,
                                                                                          device,
                                                                                          gradient_history=gradient_history,
                                                                                          reputation=reputation,
                                                                                          cos_dist=cos_dist, C=1)

                    elif args.aggregation == "signguard":
                        aggregation_rules.signguard(grad_list, net_server, args.lr, args.nbyz, byz, device,
                                                    seed=args.seed)

                    elif args.aggregation == "flare":
                        aggregation_rules.flare(grad_list, net_server, args.lr, args.nbyz, byz, device,
                                                train_server)

                    elif args.aggregation == "romoa":
                        sanitization_factor, previous_global_gradient = aggregation_rules.romoa(grad_list, net_server,
                                                                                                args.lr,
                                                                                                args.nbyz, byz,
                                                                                                device,
                                                                                                F=sanitization_factor,
                                                                                                prev_global_update=previous_global_gradient,
                                                                                                seed=args.seed)

                    else:
                        raise NotImplementedError

                    del grad_list
                    grad_list = []
                # evaluate the model accuracy
                if (global_epoch + 1) % args.test_every == 0:
                    test_metric_client_1 = predict(net_client_1, test_data, device, global_epoch, display=False, save=f"./outputs/{formatted_time}/client_1/")
                    test_metric_client_2 = predict(net_client_2, test_data, device, global_epoch, display=False, save=f"./outputs/{formatted_time}/client_2/")
                    test_metric_client_3 = predict(net_client_3, test_data, device, global_epoch, display=False, save=f"./outputs/{formatted_time}/client_3/")
                    test_metric = (test_metric_client_1+test_metric_client_2+test_metric_client_3)/3

                    test_error_list.append(test_metric)
                    test_iterations.append(global_epoch)
                    all_mse.append(test_metric)
                    print("Test the performance in Iteration %02d. Averaged MeanSquareError of all the clients %0.4f" % (global_epoch, test_metric))

        # global epochs ended

        if args.qsmc:
            server_process.wait()  # wait for process to exit

        # Append accuracy and backdoor success rate to overall runs list
        if len(runs_test_accuracy) > 0:
            runs_test_accuracy = np.vstack([runs_test_accuracy, test_error_list])
            if args.byz_type == "scaling_attack":
                runs_backdoor_success = np.vstack([runs_backdoor_success, backdoor_success_list])
        else:
            runs_test_accuracy = test_error_list
            if args.byz_type == "scaling_attack":
                runs_backdoor_success = backdoor_success_list
        if args.byz_type == "scaling_attack":
            print("Run %02d/%02d done with final accuracy: %0.4f and backdoor success rate: %0.4f" % (
            run, args.nruns, test_error_list[-1], backdoor_success_list[-1]))
        else:
            print("Run %02d/%02d done with final accuracy: %0.4f" % (run, args.nruns, test_error_list[-1]))

    del test_error_list
    test_error_list = []

    with torch.no_grad():
        print("\nSave the loss plots in ./outputs")
        plot_train_loss(client_1_losses, "client_1 losses", save=f"./outputs/{formatted_time}/client_1/")
        plot_train_loss(client_2_losses, "client_2 losses", save=f"./outputs/{formatted_time}/client_2/")
        plot_train_loss(client_3_losses, "client_3 losses", save=f"./outputs/{formatted_time}/client_3/")
        os.makedirs(f"./outputs/{formatted_time}/Server/", exist_ok=True)
        if len(server_losses)>0: plot_train_loss(server_losses, "server losses", save=f"./outputs/{formatted_time}/Server/")
        mse_client_1 = predict(net_client_1, test_data, device, "test_stage", display=False, save=f"./outputs/{formatted_time}/client_1/")
        mse_client_2 = predict(net_client_2, test_data, device, "test_stage", display=False, save=f"./outputs/{formatted_time}/client_2/")
        mse_client_3 = predict(net_client_3, test_data, device, "test_stage", display=False, save=f"./outputs/{formatted_time}/client_3/")
    print(f" Test after the final epoch:\t client_1: {mse_client_1};\t client_2: {mse_client_2};\t client_3:{mse_client_3}")

    print("all_MSE", all_mse)
    # Save the output to a text file
    with open(f"./outputs/{formatted_time}/MSE.txt", "w") as file:
        file.write(f"all_MSE: {all_mse}\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")