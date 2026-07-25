import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from util_functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.graphcnn import GraphCNN
import re
criterion = nn.CrossEntropyLoss()


# this function is to extract the test subgraph and test it to get the omla test attack
def test_subgraph(circuit_name):
    subgraph_command = "perl netlist_to_subgraphs.pl -f " + circuit_name + " -i ./circuit_datasets/" + circuit_name  #+ " > /dev/null 2>&1" 
    # results = os.popen(subgraph_command).read()
    # print(results)
    # os.system(subgraph_command)
    try:
        os.system(subgraph_command)
        print("Subgraph extracted successfully")
    except:
        print("Error in extracting the subgraph")

def test_subgraph_ori(circuit_name):
    subgraph_command = "perl netlist_to_subgraph_test.pl -f " + circuit_name + " -i ./circuit_datasets/" + circuit_name # + " > /dev/null 2>&1" 
    # results = os.popen(subgraph_command).read()
    # print(results)
    # os.system(subgraph_command)
    try:
        os.system(subgraph_command)
        print("Subgraph extracted successfully")
    except:
        print("Error in extracting the subgraph")

def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test_new(model, device, test_graphs):
    model.eval()
    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print("labels are: ", labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    print("accuracy test: %f" % (acc_test))
    # print("pred value is:", pred)
    return acc_test, labels, pred


def attack_omla_test(circuit_name, link_name, hop_size, only_predict, key_size):
    torch.manual_seed(0)
    np.random.seed(0)
    arg_device = 0
    a_list=[]
    args_only_predict = only_predict
    # args_num_layers = layer_num
    args_num_mlp_layers = 2
    args_final_dropout = 0.5
    args_graph_pooling_type = "sum"
    args_neighbor_pooling_type = "sum"
    args_learn_eps = False
    # args_hidden_dim = hidden_dimension
    args_links_name = link_name
    args_file_name = circuit_name
    device = torch.device("cuda:" + str(arg_device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    args_hop = int(hop_size)
    '''Prepare data'''
    args_file_dir = os.path.dirname(os.path.realpath('__file__'))
    val_pos, val_neg, train_pos, test_pos,train_neg,test_neg,link_pos = None,None, None, None,None,None,None
    if args_links_name is not None:
        print("The link file was provided")
        args_links_dir = os.path.join(args_file_dir, '../OMLA_' + circuit_name + '/data/{}/{}'.format(args_file_name,args_links_name))
        links_idx = np.loadtxt(args_links_dir, dtype=int)
        links_pos = (links_idx[:, 0], links_idx[:, 1])
    
    args_train_dir = os.path.join(args_file_dir, '../OMLA_' + circuit_name + '/data/{}/{}'.format(args_file_name, 'node_te_pos.txt'))
    test_pos = np.loadtxt(args_train_dir, dtype=int)

    test_idxx = np.loadtxt(args_train_dir, dtype=int)
    a_list.append(test_idxx)

    args_train_dir = os.path.join(args_file_dir, '../OMLA_' + circuit_name + '/data/{}/{}'.format(args_file_name, 'node_te_neg.txt'))
    test_neg = np.loadtxt(args_train_dir, dtype=int)

    test_idx2 = np.loadtxt(args_train_dir, dtype=int)
    a_list.append(test_idx2)
    test_idx=np.concatenate(a_list)

    print("All done!")
    cell=[]
    feat=[]
    count=[]
    feats_test = np.loadtxt('../OMLA_' + circuit_name + '/data/{}/feat.txt'.format(args_file_name), dtype='float32')
    count = np.loadtxt('../OMLA_' + circuit_name + '/data/{}/count.txt'.format(args_file_name))
    cell = np.genfromtxt('../OMLA_' + circuit_name + '/data/{}/cell.txt'.format(args_file_name), dtype=str)
    arr1inds = count.argsort()
    sorted_count = count[arr1inds[0::]]
    attributes = feats_test[arr1inds[0::]]
    sorted_cell = cell[arr1inds[0::]]

    max_idx = np.max(links_idx)
    net = ssp.csc_matrix((np.ones(len(links_idx)), (links_idx[:, 0], links_idx[:, 1])), shape=(max_idx+1, max_idx+1) )

    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    B=net.copy() # a matrix without semmetric edges
    B.eliminate_zeros()
    net[links_idx[:, 1], links_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

    node_information = attributes
    args_no_parallel = False
    args_use_dis = True

    if args_only_predict:  
        print("Inside the only predict function")

        _, test_graphs,_ = keygates2subgraphs(
            A,
            B,
            None,
            None,
            test_pos,
            test_neg,
            None,
            None,
            args_hop,
            node_information,
            args_no_parallel,
            args_use_dis
        )
        print('# test: %d' % (len(test_graphs)))
    else:
        train_graphs, test_graphs,val_graphs = keygates2subgraphs(
            A,
            B,
            train_pos,
            train_neg,
            test_pos,
            test_neg,
            val_pos,
            val_neg,
            args_hop,
            node_information,
            args_no_parallel,
            args_use_dis
        )
        print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    num_classes=2
    if args_only_predict:
        args_data_name='model'
        print("We are predicting")
        with open('../OMLA_' + circuit_name + '/data/{}/{}_hyper.pkl'.format(args_file_name,args_data_name), 'rb') as hyperparameters_name:
            saved_args = pickle.load(hyperparameters_name)
        
        args_num_layers = vars(saved_args)['num_layers']
        args_hidden_dim = vars(saved_args)['hidden_dim']
        # for key, value in vars(saved_args).items(): # replace with saved cmd_args
        #     # vars(args)[key] = value
        #     print(key)
        #     print(value)
        # # Update local variables with saved hyperparameters
        # args_num_layers = saved_args.get('num_layers', args_num_layers)  # default_num_layers should be defined elsewhere
        # args_num_mlp_layers = saved_args.get('num_mlp_layers', args_num_mlp_layers)
        # args_hidden_dim = saved_args.get('hidden_dim', args_hidden_dim)
        # args_final_dropout = saved_args.get('final_dropout', args_final_dropout)
        # args_learn_eps = saved_args.get('learn_eps', args_learn_eps)
        # args_graph_pooling_type = saved_args.get('graph_pooling_type', args_graph_pooling_type)
        # args_neighbor_pooling_type = saved_args.get('neighbor_pooling_type', args_neighbor_pooling_type)
        
        classifier = GraphCNN(args_num_layers, args_num_mlp_layers, test_graphs[0].node_features.shape[1], args_hidden_dim, num_classes, args_final_dropout, args_learn_eps, args_graph_pooling_type, args_neighbor_pooling_type, device).to(device)
        if torch.cuda.is_available():
            classifier = classifier.cuda()
        model_name = '../OMLA_' + circuit_name + '/data/{}/{}_model.pth'.format(args_file_name,args_data_name)
        classifier.load_state_dict(torch.load(model_name))
        acc_test, labels, predictions = test_new(classifier, device,test_graphs)

        new_predictions=predictions.reshape(labels.shape[0],1)
        new_labels=labels.reshape(labels.shape[0],1)
        new_test_idx=test_idx.reshape(test_idx.shape[0],1)
        
        test_idx_and_pred = np.concatenate([new_test_idx, new_predictions, new_labels],1)
        pred_name = '../OMLA_' + circuit_name + '/data/{}/'.format(args_file_name) + 'h'+str(args_hop)+'_pred.txt'
        np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%1.2f', '%d'])
        print('Predictions for {} are saved in {}'.format(args_file_name, pred_name))
        # based on the new_test_idx to find out the keyinputname and sort them
        keyinput_pred = {}
        keyinput_label = {}
        for index in range(len(new_test_idx)):
            item = new_test_idx[index]
            if item[0] < 100:
                gatename = "g0"+str(item[0])
            else:
                gatename = "g"+str(item[0])
            with open("../OMLA_" + circuit_name + "/" + circuit_name + "/Test_"+circuit_name.split(str(key_size))[0]+"_syn_locked_rnd_" + str(key_size) + "_syn.v", "r") as f:
                for line in f:
                    if gatename in line:
                        regex = r"KEYINPUT(\d+)"
                        matches = re.findall(regex, line)
                        keyinputs = [int(num) for num in matches]
                        for keyinput in keyinputs: 
                            if keyinput not in keyinput_pred.keys():
                                keyinput_pred[keyinput] = [new_predictions[index][0].item()]
                                keyinput_label[keyinput] = [new_labels[index][0].item()]
                            else:
                                keyinput_pred[keyinput].append(new_predictions[index][0].item())
        # sort the keyinput_pred based on its key
        keyinput_pred = dict(sorted(keyinput_pred.items()))
        keyinput_label = dict(sorted(keyinput_label.items()))
        # print("keyinput_pred is: ", keyinput_pred)
        # print("keyinput_label is: ", keyinput_label)
        correct_num = 0
        wrong_num = 0
        X_num = 0
        for key, value in keyinput_pred.items():
            if len(value) == 1:
                if value[0] == keyinput_label[key][0]:
                    correct_num += 1
                else:
                    wrong_num += 1
            else:
                # get the 1/0 number in the list
                one_num = value.count(1)
                zero_num = value.count(0)
                if one_num > zero_num:
                    if 1 in keyinput_label[key]:
                        correct_num += 1
                    else:
                        wrong_num += 1
                elif one_num < zero_num:
                    if 0 in keyinput_label[key]:
                        correct_num += 1
                    else:
                        wrong_num += 1
                else:
                    X_num += 1
        # print("correct_num is: ", correct_num)
        # print("wrong_num is: ", wrong_num)
        # print("X_num is: ", X_num)
        # print("total", correct_num + wrong_num + X_num)
        acc_test = correct_num / len(keyinput_label)
        pred_test = (correct_num + X_num)/len(keyinput_label)
        kpa_test = correct_num / (len(keyinput_label) - X_num)
        print("acc_test is: ", acc_test)
        print("pred_test is: ", pred_test)
        print("kpa_test is: ", kpa_test)

        # exit()
    return acc_test, pred_test, kpa_test

def attack_omla_test_ori(circuit_name, link_name, hop_size, only_predict, key_size):
    torch.manual_seed(0)
    np.random.seed(0)
    arg_device = 0
    a_list=[]
    args_only_predict = only_predict
    # args_num_layers = layer_num
    args_num_mlp_layers = 2
    args_final_dropout = 0.5
    args_graph_pooling_type = "sum"
    args_neighbor_pooling_type = "sum"
    args_learn_eps = False
    # args_hidden_dim = hidden_dimension
    args_links_name = link_name
    args_file_name = circuit_name
    device = torch.device("cuda:" + str(arg_device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    args_hop = int(hop_size)
    '''Prepare data'''
    args_file_dir = os.path.dirname(os.path.realpath('__file__'))
    val_pos, val_neg, train_pos, test_pos,train_neg,test_neg,link_pos = None,None, None, None,None,None,None
    if args_links_name is not None:
        print("The link file was provided")
        args_links_dir = os.path.join(args_file_dir, './data/{}/{}'.format(args_file_name,args_links_name))
        links_idx = np.loadtxt(args_links_dir, dtype=int)
        links_pos = (links_idx[:, 0], links_idx[:, 1])
    
    args_train_dir = os.path.join(args_file_dir, './data/{}/{}'.format(args_file_name, 'node_te_pos.txt'))
    test_pos = np.loadtxt(args_train_dir, dtype=int)

    test_idxx = np.loadtxt(args_train_dir, dtype=int)
    a_list.append(test_idxx)

    args_train_dir = os.path.join(args_file_dir, './data/{}/{}'.format(args_file_name, 'node_te_neg.txt'))
    test_neg = np.loadtxt(args_train_dir, dtype=int)

    test_idx2 = np.loadtxt(args_train_dir, dtype=int)
    a_list.append(test_idx2)
    test_idx=np.concatenate(a_list)

    print("All done!")
    cell=[]
    feat=[]
    count=[]
    feats_test = np.loadtxt('./data/{}/feat.txt'.format(args_file_name), dtype='float32')
    count = np.loadtxt('./data/{}/count.txt'.format(args_file_name))
    cell = np.genfromtxt('./data/{}/cell.txt'.format(args_file_name), dtype=str)
    arr1inds = count.argsort()
    sorted_count = count[arr1inds[0::]]
    attributes = feats_test[arr1inds[0::]]
    sorted_cell = cell[arr1inds[0::]]

    max_idx = np.max(links_idx)
    net = ssp.csc_matrix((np.ones(len(links_idx)), (links_idx[:, 0], links_idx[:, 1])), shape=(max_idx+1, max_idx+1) )

    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    B=net.copy() # a matrix without semmetric edges
    B.eliminate_zeros()
    net[links_idx[:, 1], links_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

    node_information = attributes
    args_no_parallel = False
    args_use_dis = True

    if args_only_predict:  
        print("Inside the only predict function")

        _, test_graphs,_ = keygates2subgraphs(
            A,
            B,
            None,
            None,
            test_pos,
            test_neg,
            None,
            None,
            args_hop,
            node_information,
            args_no_parallel,
            args_use_dis
        )
        print('# test: %d' % (len(test_graphs)))
    else:
        train_graphs, test_graphs,val_graphs = keygates2subgraphs(
            A,
            B,
            train_pos,
            train_neg,
            test_pos,
            test_neg,
            val_pos,
            val_neg,
            args_hop,
            node_information,
            args_no_parallel,
            args_use_dis
        )
        print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    num_classes=2
    if args_only_predict:
        args_data_name='model'
        print("We are predicting")
        with open('./data/{}/{}_hyper.pkl'.format(args_file_name,args_data_name), 'rb') as hyperparameters_name:
            saved_args = pickle.load(hyperparameters_name)
        # for key, value in vars(saved_args).items(): # replace with saved cmd_args
            # vars(args)[key] = value
        args_num_layers = vars(saved_args)['num_layers']
        args_hidden_dim = vars(saved_args)['hidden_dim']
        print(args_num_layers, args_hidden_dim, vars(saved_args)['batch_size'])
        # # Update local variables with saved hyperparameters
        # args_num_layers = saved_args.get('num_layers', args_num_layers)  # default_num_layers should be defined elsewhere
        # args_num_mlp_layers = saved_args.get('num_mlp_layers', args_num_mlp_layers)
        # args_hidden_dim = saved_args.get('hidden_dim', args_hidden_dim)
        # args_final_dropout = saved_args.get('final_dropout', args_final_dropout)
        # args_learn_eps = saved_args.get('learn_eps', args_learn_eps)
        # args_graph_pooling_type = saved_args.get('graph_pooling_type', args_graph_pooling_type)
        # args_neighbor_pooling_type = saved_args.get('neighbor_pooling_type', args_neighbor_pooling_type)
        
        classifier = GraphCNN(args_num_layers, args_num_mlp_layers, test_graphs[0].node_features.shape[1], args_hidden_dim, num_classes, args_final_dropout, args_learn_eps, args_graph_pooling_type, args_neighbor_pooling_type, device).to(device)
        if torch.cuda.is_available():
            classifier = classifier.cuda()
        model_name = './data/{}/{}_model.pth'.format(args_file_name,args_data_name)
        classifier.load_state_dict(torch.load(model_name))
        acc_test, labels, predictions = test_new(classifier, device,test_graphs)

        new_predictions=predictions.reshape(labels.shape[0],1)
        new_labels=labels.reshape(labels.shape[0],1)
        new_test_idx=test_idx.reshape(test_idx.shape[0],1)
        
        test_idx_and_pred = np.concatenate([new_test_idx, new_predictions, new_labels],1)
        pred_name = './data/{}/'.format(args_file_name) + 'h'+str(args_hop)+'_pred.txt'
        np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%1.2f', '%d'])
        print('Predictions for {} are saved in {}'.format(args_file_name, pred_name))
        # based on the new_test_idx to find out the keyinputname and sort them
        # keyinput_pred = {}
        # keyinput_label = {}
        # gate_regex = 'g0*(\d+)'

        # # with open("../OMLA_" + circuit_name + "/" + circuit_name + "/Test_"+circuit_name + "_syn_locked_rnd_" + str(key_size) + "_syn.v", "r") as f:
        # with open("circuit_datasets/" + circuit_name + "/Test_"+ "c6288" + "_syn_locked_rnd_" + str(key_size) + "_syn.v", "r") as f:
        #     for line in f:
        #         gate_id = re.findall(gate_regex, line)
        #         if len(gate_id) == 0:
        #             continue
        #         else:
        #             # print("what is gate_id:", gate_id)
        #             gate_id = gate_id[0]
        #         regex = r"KEYINPUT(\d+)"
        #         matches = re.findall(regex, line)
        #         keyinputs = [int(num) for num in matches]
        #         if [int(gate_id)] in new_test_idx:
        #             index = np.argmax(new_test_idx == [int(gate_id)])
        #             for keyinput in keyinputs: 
        #                 if keyinput not in keyinput_pred.keys():
        #                     keyinput_pred[keyinput] = [new_predictions[index][0].item()]
        #                     keyinput_label[keyinput] = [new_labels[index][0].item()]
        #                 else:
        #                     keyinput_pred[keyinput].append(new_predictions[index][0].item())

        # # sort the keyinput_pred based on its key
        # keyinput_pred = dict(sorted(keyinput_pred.items()))
        # keyinput_label = dict(sorted(keyinput_label.items()))
        # print("keyinput_pred is: ", keyinput_pred)
        # print("keyinput_label is: ", keyinput_label)
        # correct_num = 0
        # wrong_num = 0
        # X_num = 0
        # for key, value in keyinput_pred.items():
        #     if len(value) == 1:
        #         if value[0] == keyinput_label[key][0]:
        #             correct_num += 1
        #         else:
        #             wrong_num += 1
        #     else:
        #         # get the 1/0 number in the list
        #         one_num = value.count(1)
        #         zero_num = value.count(0)
        #         if one_num > zero_num:
        #             if 1 in keyinput_label[key]:
        #                 correct_num += 1
        #             else:
        #                 wrong_num += 1
        #         elif one_num < zero_num:
        #             if 0 in keyinput_label[key]:
        #                 correct_num += 1
        #             else:
        #                 wrong_num += 1
        #         else:
        #             X_num += 1
        # # print("correct_num is: ", correct_num)
        # # print("wrong_num is: ", wrong_num)
        # # print("X_num is: ", X_num)
        # # print("total", correct_num + wrong_num + X_num)
        # acc_test = correct_num / len(keyinput_label)
        # pred_test = (correct_num + X_num)/len(keyinput_label)
        # kpa_test = correct_num / (len(keyinput_label) - X_num)
        # print("acc_test is: ", acc_test)
        # print("pred_test is: ", pred_test)
        # print("kpa_test is: ", kpa_test)

        # exit()
    return acc_test#, pred_test, kpa_test


# test_subgraph("c1355")

# print("acc_test is: ", acc_test)
# get the omla test accuracy
def get_omla_key_acc(circuit_name, link_name, hop_size, only_predict, key_size):
    # log_file = circuit_name + "_log.txt"
    # sys.stdout = log_file
    test_subgraph(circuit_name)
    # log_file = "../OMLA_" + circuit_name + "/" + circuit_name + "_test_log.txt"
    # sys.stdout = log_file
    acc_test = attack_omla_test(circuit_name, link_name, hop_size, only_predict, key_size)
    # sys.stdout = sys.__stdout__
    # log_file.close()
    return acc_test

def get_omla_key_acc_ori(circuit_name, link_name, hop_size, only_predict, key_size):
    # log_file = circuit_name + "_log.txt"
    # sys.stdout = log_file
    test_subgraph(circuit_name)
    # log_file = "../OMLA_" + circuit_name + "/" + circuit_name + "_test_log.txt"
    # sys.stdout = log_file
    acc_test = attack_omla_test_ori(circuit_name, link_name, hop_size, only_predict, key_size)
    # sys.stdout = sys.__stdout__
    # log_file.close()
    return acc_test

# check the file name 
def change_file_name(circuit_name):
    folder_path = "circuit_datasets/" + circuit_name
    file_list = os.listdir(folder_path)
    Train_file_list = []
    for file_name in file_list:
        if "Train" in file_name:
            Train_file_list.append(file_name)
    # split 9:1 to train and validate, last 10% for validate
    Train_file_list.sort()
    # last 10% files
    Train_file_list = Train_file_list[-int(len(Train_file_list)/10):]
    for file_name in Train_file_list:
        os.rename(folder_path + "/" + file_name, folder_path + "/" + file_name.replace("Train", "Validate"))

    
    


# acc_test = get_omla_key_acc_ori("c43220", "link.txt", 2, True, 32)
# acc_test = get_omla_key_acc_ori("c49920", "link.txt", 2, True, 32)
# acc_test = get_omla_key_acc_ori("c88020", "link.txt", 2, True, 64)
# acc_test = get_omla_key_acc_ori("c135520", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c190820", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c267020", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c354020", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c628820", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c755220", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c3540128", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c5315128", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c6288128", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c7552128", "link.txt", 2, True, 128)
# acc_test = get_omla_key_acc_ori("c3540_test", "link.txt", 1, True, 64)
# print("acc_test is: ", acc_test)
# test_subgraph_ori("decoder20")
# change_file_name("b1920")