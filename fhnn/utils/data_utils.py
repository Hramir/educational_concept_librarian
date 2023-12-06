"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
from typing import List
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import json
from utils.constants_utils import NUM_TRANSCRIPTS, MAX_CONCEPT_HIERARCHY_SIZE

THICK_INDEX = 0
MYELIN_INDEX = 1
TREE_DEPTH = 10
THRESHOLD = 0.07
NUM_ROIS = 360
THRESHOLD_NEW = 0.23
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = .7, .2, .1

NUM_MEG_COLE_ROIS = 90
MEG_PLV_METRIC_INDEX = 0
MEG_BAND_ALPHA_INDEX = 1
THRESHOLD_MEG = 0.329
NUM_SBJS_MEG = 180

TOTAL_SBJS = 592
NUM_SBJS = TOTAL_SBJS - 5 # 587 subjects after filtering out numerically unstable PLV matrices
# Remember that 5 subjects excluded due to filtering out Numerically Unstable PLV Matrices

def load_data(args, datapath):
    if args.dataset == 'library_learning':
        train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = load_data_library_learning(args.dataset, args.use_feats, datapath)
        data = {'train_graph_data_dicts' : train_graph_data_dicts, 
                'val_graph_data_dicts' : val_graph_data_dicts,
                'test_graph_data_dicts'  : test_graph_data_dicts,
        }

        for graph_data_dicts in data.values():
            for graph_data in graph_data_dicts:
                adj = graph_data['adj_train']
                
                adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                        adj, args.val_prop, args.test_prop, args.split_seed
                )
                graph_data['adj_train'] = adj_train
                graph_data['train_edges'], graph_data['train_edges_false'] = train_edges, train_edges_false
                graph_data['val_edges'], graph_data['val_edges_false'] = val_edges, val_edges_false
                graph_data['test_edges'], graph_data['test_edges_false'] = test_edges, test_edges_false
                
                graph_data['edges'] = train_edges 
                graph_data['edges_false'] = train_edges_false
                graph_data['adj_train_norm'], graph_data['features'] = process(
                    graph_data['adj_train'], graph_data['features'], args.normalize_adj, args.normalize_feats
                )
        return data

    if args.dataset == 'cam_can_multiple' or args.dataset == 'binary_cyclic_tree_multiple' or args.dataset == 'meg':
        data = load_data_lp(args.dataset, args.use_feats, datapath, args.use_super_node, args.use_thicks_myelins, threshold = args.threshold)

        for graph_data_dicts in data.values():
            for graph_data in graph_data_dicts:
                adj = graph_data['adj_train']
                
                adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                        adj, args.val_prop, args.test_prop, args.split_seed
                )
                graph_data['adj_train'] = adj_train
                graph_data['train_edges'], graph_data['train_edges_false'] = train_edges, train_edges_false
                graph_data['val_edges'], graph_data['val_edges_false'] = val_edges, val_edges_false
                graph_data['test_edges'], graph_data['test_edges_false'] = test_edges, test_edges_false
                
                graph_data['edges'] = train_edges 
                graph_data['edges_false'] = train_edges_false
                graph_data['adj_train_norm'], graph_data['features'] = process(
                    graph_data['adj_train'], graph_data['features'], args.normalize_adj, args.normalize_feats
                )
        return data

    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath, args.use_super_node, threshold = args.threshold)
        
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
def z_score_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # Training with 100 % edges visible since will do graph iteration and train, val, test splits will come from graphs
    train_edges = pos_edges
    train_edges_false = neg_edges

    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################
def add_super_node_to_adjacency_matrix(adj):
    adj_matrix = np.zeros((adj.shape[0] + 1, adj.shape[1] + 1))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            adj_matrix[i][j] = adj[i][j]
    for i in range(len(adj_matrix)):
        adj_matrix[i][-1] = 1
    for j in range(len(adj_matrix[0])):
        adj_matrix[-1][j] = 1
    adj_matrix[-1][-1] = 0
    return adj_matrix

def load_data_lp(dataset : str, use_feats : bool, data_path : str, use_super_node : bool =False, use_thicks_myelins : bool =True, threshold : float =1000.0):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset == 'binary_tree':
        def get_adjacency_matrix_for_binary_tree(depth):
            adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])
            
            for node_index in range(1, (2 ** depth - 1) // 2 + 1):
                adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
                adj_matrix[node_index - 1][2 * node_index] = 1.0
                adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
                adj_matrix[2 * node_index][node_index - 1] = 1.0
                
            return adj_matrix
        
        adj_mat = get_adjacency_matrix_for_binary_tree(TREE_DEPTH)
        adj, features = sp.csr_matrix(adj_mat), np.eye(len(adj_mat))
    elif dataset == 'binary_cyclic_tree':
        if use_super_node: adj_mat = get_adjacency_matrix_for_binary_cyclic_tree_with_super_node(TREE_DEPTH)
        else: adj_mat = get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) 
        adj, features = sp.csr_matrix(adj_mat), np.eye(len(adj_mat))

        def create_permutation(permutation_length):
            permutation = []
            seen = set()
            for i in range(permutation_length):
                if i == 0: num = np.random.randint(0, permutation_length)
                while num in seen:
                    num = np.random.randint(0, permutation_length)
                seen.add(num)
                permutation.append(num)
            return permutation
        def create_permutation_matrix(permutation):
            n = len(permutation)
            permutation_matrix = np.eye(n)
            
            # Rearrange rows/columns based on the permutation
            for i in range(n):
                j = permutation[i]
                if i != j:
                    permutation_matrix[[i, j]] = permutation_matrix[[j, i]]
            
            return permutation_matrix
        use_permutation = False
        if use_permutation:
            permutation = create_permutation(len(adj_mat))
            features = create_permutation_matrix(permutation)
        use_random_noise = False
        if use_random_noise:
            print("USING RANDOM NOISE")
            noise = np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat)))
            print("NOISE: ", noise)
            features = features + noise
        # Reverse Identity Matrix
        # features = np.flipud(np.diag(np.ones(len(adj_mat))))
        # features = np.eye(len(adj_mat))
        # print("PERMUTATION FEATURES", permutation)
        # print("PERMUTATION FEATURES", features[:5, :5])
        import logging
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Feature Matrix: {features}")
        
    elif dataset == 'cam_can_avg':
        # ADJ MAT
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_276_sbj.npy"))
        plv_average = sum(plv_tensor) / len(plv_tensor)
        plv_node_features = plv_average.copy()
        plv_matrix = plv_average.copy()
        plv_matrix [plv_matrix >= THRESHOLD] = 1
        plv_matrix [plv_matrix < THRESHOLD] = 0
        
        adj = plv_matrix.copy()
        
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        # Thicks Myelins Features is 2 x 276 x 360
        # FEATURES 276 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor.npy")) # 276 as well  
            
            average_thicks_myelins = np.sum(thicks_myelins_tensor, axis = 1) / len(thicks_myelins_tensor[0])
            
            thicks_myelins_features = average_thicks_myelins.T
            use_normalize_thicks_myelins = False
            
            if use_normalize_thicks_myelins: 
                thicks_avg = thicks_myelins_features[:, THICK_INDEX]
                myelins_avg = thicks_myelins_features[:, MYELIN_INDEX]
                thicks_avg_mean = np.mean(thicks_avg)
                thicks_avg_sd = np.std(thicks_avg)
                myelins_avg_mean = np.mean(myelins_avg)
                myelins_avg_sd = np.std(myelins_avg)
                normalized_thicks_avg = [(thick - thicks_avg_mean) / thicks_avg_sd for thick in thicks_avg]
                normalized_myelins_avg = [(myelin - myelins_avg_mean) / myelins_avg_sd for myelin in myelins_avg]
                thicks_myelins_features = np.vstack((normalized_thicks_avg, normalized_myelins_avg)), (NUM_ROIS, 2).T
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
                
        else:
            if use_super_node: 
                features = np.eye(NUM_ROIS + 1)
            else: 
                features = np.eye(NUM_ROIS)
        
        
        adj = sp.csr_matrix(adj_mat)
    elif dataset == 'cam_can_avg_new':
        use_noise = False
        # (592 -5) x 360 x 360
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
        plv_tensor = remove_self_loops(plv_tensor)
        plv_average = np.sum(plv_tensor, axis=0) / len(plv_tensor)
        plv_node_features = plv_average.copy()
        # ADJ MAT
        plv_matrix = plv_average.copy()
        if use_noise: plv_matrix = plv_matrix + np.random.randn(*plv_matrix.shape)

        plv_matrix [plv_matrix >= THRESHOLD_NEW] = 1
        plv_matrix [plv_matrix < THRESHOLD_NEW] = 0
        
        adj = plv_matrix.copy()
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        # Thicks Myelins Features is 2 x 592 x 360
        # FEATURES 592 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
            
            
            average_thicks_myelins = np.sum(thicks_myelins_tensor, axis = 1) / len(thicks_myelins_tensor[0])
            
            thicks_myelins_features = average_thicks_myelins.T
            
            use_normalize_thicks_myelins = False
            if use_normalize_thicks_myelins: 
                thicks_avg = thicks_myelins_features[:, THICK_INDEX]
                myelins_avg = thicks_myelins_features[:, MYELIN_INDEX]
                thicks_avg_mean = np.mean(thicks_avg)
                thicks_avg_sd = np.std(thicks_avg)
                myelins_avg_mean = np.mean(myelins_avg)
                myelins_avg_sd = np.std(myelins_avg)
                normalized_thicks_avg = [(thick - thicks_avg_mean) / thicks_avg_sd for thick in thicks_avg]
                normalized_myelins_avg = [(myelin - myelins_avg_mean) / myelins_avg_sd for myelin in myelins_avg]

                thicks_myelins_features = np.vstack((normalized_thicks_avg, normalized_myelins_avg)).T
            
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
        else:
            if use_super_node: features = np.eye(NUM_ROIS + 1)
            else: features = np.eye(NUM_ROIS)
        adj = sp.csr_matrix(adj_mat)
        
        if use_noise: features = features + np.random.randn(*features.shape) * 1 / 100

    elif dataset == 'cam_can_single':
        # ADJ MAT
        subject_index = 275
        age_label = np.load(os.path.join(data_path, "age_labels_276_sbj.npy"))[subject_index]
        print(f"Subject Index : {subject_index}")
        print(f"Subject Age : {age_label}")
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_276_sbj.npy"))
        plv_sbj_matrix = plv_tensor[subject_index] 
        plv_node_features = plv_sbj_matrix.copy()
        plv_matrix = plv_sbj_matrix.copy()
        plv_matrix [plv_matrix >= THRESHOLD] = 1
        plv_matrix [plv_matrix < THRESHOLD] = 0
        
        adj = plv_matrix.copy()
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        # Thicks Myelins is 2 x 276 x 360
        # FEATURES 276 x 2 x 360
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor.npy")) # 276 as well  
            
            
            thicks_sbj = thicks_myelins_tensor[THICK_INDEX][subject_index]
            myelins_sbj = thicks_myelins_tensor[MYELIN_INDEX][subject_index]
            use_normalize_thicks_myelins = False
            if use_normalize_thicks_myelins: 
                thicks_sbj_mean = np.mean(thicks_sbj)
                thicks_sbj_sd = np.std(thicks_sbj)
                myelins_sbj_mean = np.mean(myelins_sbj)
                myelins_sbj_sd = np.std(myelins_sbj)
                normalized_thicks_sbj = [(thick - thicks_sbj_mean) / thicks_sbj_sd for thick in thicks_sbj]
                normalized_myelins_sbj = [(myelin - myelins_sbj_mean) / myelins_sbj_sd for myelin in myelins_sbj]
                thicks_sbj, myelins_sbj = normalized_thicks_sbj, normalized_myelins_sbj    

            thicks_myelins_features = np.vstack((thicks_sbj, myelins_sbj)).T
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
        else:
            if use_super_node: features = np.eye(NUM_ROIS + 1) 
            else: features = np.eye(NUM_ROIS)
        adj = sp.csr_matrix(adj_mat)
        
    elif dataset == 'cam_can_single_new':
        # ADJ MAT
        subject_index = 100
        age_label = np.load(os.path.join(data_path, "age_labels_592_sbj_filtered.npy"))[subject_index]
        print(f"Subject Index : {subject_index}")
        print(f"Subject Age : {age_label}")
        plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
        plv_tensor = remove_self_loops(plv_tensor)
        plv_sbj_matrix = plv_tensor[subject_index] 
        plv_node_features = plv_sbj_matrix.copy()
        plv_matrix_thresholded = plv_sbj_matrix.copy()
        plv_matrix_thresholded [plv_matrix_thresholded >= THRESHOLD_NEW] = 1
        plv_matrix_thresholded [plv_matrix_thresholded < THRESHOLD_NEW] = 0
        
        adj = plv_matrix_thresholded.copy()
        if use_super_node: adj_mat = add_super_node_to_adjacency_matrix(adj)
        else: adj_mat = adj
        
        # Thicks Myelins is 2 x 276 x 360
        # FEATURES 276 x 2 x 360
        print("USING IDENTITY MATRIX")
        use_feats = False
        if use_feats:
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 276 as well  
            
            thicks_sbj = thicks_myelins_tensor[THICK_INDEX][subject_index]
            myelins_sbj = thicks_myelins_tensor[MYELIN_INDEX][subject_index]
            
            use_min_max_scaler = True
            use_z_score_normalization = False
            if use_min_max_scaler:
                thicks_sbj = min_max_normalize(thicks_sbj)
                myelins_sbj = min_max_normalize(myelins_sbj)    
                plv_node_features = min_max_normalize(plv_node_features)
            if use_z_score_normalization:
                thicks_sbj = z_score_normalize(thicks_sbj)
                myelins_sbj = z_score_normalize(myelins_sbj)
            
            thicks_myelins_features = np.vstack((thicks_sbj, myelins_sbj)).T
            features = np.hstack((thicks_myelins_features, plv_node_features))
            if use_super_node: 
                one_hot_vec_for_super_node = np.zeros(len(features[0]))
                one_hot_vec_for_super_node[-1] = 1
                features = np.vstack((features, one_hot_vec_for_super_node))
        else:
            if use_super_node: features = np.eye(NUM_ROIS + 1) 
            else: features = np.eye(NUM_ROIS)
        adj = sp.csr_matrix(adj_mat)
    elif dataset == 'cam_can_multiple':
        train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = load_data_cam_can_new(threshold, NUM_SBJS, data_path, use_thicks_myelins, use_super_node)
    elif dataset == 'meg':
        
        train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = load_data_meg(NUM_SBJS_MEG, data_path, use_thicks_myelins, use_super_node)
    elif dataset == 'binary_cyclic_tree_multiple':
        train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = load_data_binary_cyclic_tree_multiple(NUM_SBJS, data_path)

    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    if dataset =='cam_can_multiple' or dataset == "binary_cyclic_tree_multiple" or dataset == "meg":
        data = {'train_graph_data_dicts' : train_graph_data_dicts, 
                'val_graph_data_dicts' : val_graph_data_dicts,
                'test_graph_data_dicts'  : test_graph_data_dicts,
        }

    else:
        data = {'adj_train': adj, 'features': features}
    
    return data

def load_data_library_learning(data_path : str, use_thicks_myelins : bool, use_super_node : bool):
    # ADJ MAT
    # 360 x 360
    # FEATURES 592 x 2 x 360
    # Thicks Myelins is 2 x 592 x 360

    # 360 x 2 n_nodes = 360 feat_dim = 2
    # 360 x 362 n_nodes = 360 feat_dim = 362

    [train_adj_matrices, 
        train_feats, 
        train_age_labels,
        train_split_indices,
        val_adj_matrices, 
        val_feats,
        val_age_labels,
        val_split_indices, 
        test_adj_matrices, 
        test_feats,
        test_age_labels,
        test_split_indices] = get_adj_mat_dataset_splits_with_features_and_score_labels_and_indices(data_path)
    
    train_graph_data_dicts = [{'adj_train': sp.csr_matrix(train_adj), 
                                'features': train_feat,
                                'age_label': train_age_label,
                                'index': train_split_index} 
                                for train_adj, train_feat, train_age_label, train_split_index 
                                    in zip(train_adj_matrices, train_feats, train_age_labels, train_split_indices)]
    val_graph_data_dicts = [{'adj_train': sp.csr_matrix(val_adj), 
                                'features': val_feat,
                                'age_label': val_age_label,
                                'index': val_split_index} 
                                for val_adj, val_feat, val_age_label, val_split_index 
                                    in zip(val_adj_matrices, val_feats, val_age_labels, val_split_indices)]
    test_graph_data_dicts = [{'adj_train': sp.csr_matrix(test_adj), 
                                'features': test_feat,
                                'age_label': test_age_label,
                                'index': test_split_index} 
                                for test_adj, test_feat, test_age_label, test_split_index 
                                    in zip(test_adj_matrices, test_feats, test_age_labels, test_split_indices)]

    return train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts

def get_adj_mat_dataset_splits_with_features_and_score_labels_and_indices(data_path : str):
    
    train_num, val_num = int(NUM_TRANSCRIPTS * TRAIN_SPLIT), int(NUM_TRANSCRIPTS * VAL_SPLIT)
    test_num = NUM_TRANSCRIPTS - train_num - val_num 
    assert train_num + val_num + test_num == NUM_TRANSCRIPTS, "Must add to total number of subjects"
    train_split_indices, val_split_indices, test_split_indices = get_dataset_split_indices(NUM_TRANSCRIPTS)
    # NOTE: Using subset of data for now to test and fix clumping issue, will use full dataset once clumping issue is fixed
    # UPDATE: Seem to have fixed issue! Now using full dataset
    # train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]
    # import logging
    # logging.info("Using 5 Graphs")
    # train_split_indices, val_split_indices, test_split_indices = [100, 100, 100, 100, 100], [100, 100, 100], [100, 100]
    # train_split_indices, val_split_indices, test_split_indices = [100, 100, 100], [100], [100]
    # train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]
    

    # TODO: Remove this once hierarchy testing is complete!
    # Must ensure same adjacency matrix with additional edges fed into model for each split to test hierarchy changes
    # logging.info("Using same adjacency matrix to test Hierarchy Changes!")
    # adjacency_matrix = get_adjacency_matrix_cam_can(100, data_path, use_super_node)
    # train_split_adj_matrices = [adjacency_matrix for _ in train_split_indices]
    # val_split_adj_matrices = [adjacency_matrix for _ in val_split_indices]
    # test_split_adj_matrices = [adjacency_matrix for _ in test_split_indices]
    print("THESE ARE THE TRAINING INDEICES", train_split_indices)
    train_split_adj_matrices = [get_adjacency_matrix_library_learning(i, data_path) for i in train_split_indices]
    val_split_adj_matrices = [get_adjacency_matrix_library_learning(i, data_path) for i in val_split_indices]
    test_split_adj_matrices = [get_adjacency_matrix_library_learning(i, data_path) for i in test_split_indices]
    train_feats = [np.eye(MAX_CONCEPT_HIERARCHY_SIZE) for train_adj_matrix in train_split_adj_matrices]
    val_feats = [np.eye(MAX_CONCEPT_HIERARCHY_SIZE) for val_adj_matrix in val_split_adj_matrices]
    test_feats = [np.eye(MAX_CONCEPT_HIERARCHY_SIZE) for test_adj_matrix in test_split_adj_matrices]
    
    # train_feats = [get_feature_matrix_library_learning(i, data_path) for i in train_split_indices]
    # val_feats = [get_feature_matrix_library_learning(i, data_path) for i in val_split_indices]
    # test_feats = [get_feature_matrix_library_learning(i, data_path) for i in test_split_indices]
    # TODO: FIX SCORE_LABELS
    import pandas as pd
    transcript_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'library_learning', "video_transcripts_with_hierarchy_1701550479.csv"))
    view_count_labels = transcript_df['view_count']
    score_labels = view_count_labels
    # score_labels = np.load(os.path.join(data_path, "score_labels.npy"))
    train_score_labels = [score_labels[i] for i in train_split_indices]
    val_score_labels = [score_labels[i] for i in val_split_indices]
    test_score_labels = [score_labels[i] for i in test_split_indices]

    return [train_split_adj_matrices, 
            train_feats, 
            train_score_labels,
            train_split_indices, 
            val_split_adj_matrices, 
            val_feats, 
            val_score_labels,
            val_split_indices,
            test_split_adj_matrices, 
            test_feats, 
            test_score_labels,
            test_split_indices]

def load_data_cam_can_new(threshold : float, num_sbjs : int, data_path : str, use_thicks_myelins : bool, use_super_node : bool):
    # ADJ MAT
    # 360 x 360
    # FEATURES 592 x 2 x 360
    # Thicks Myelins is 2 x 592 x 360

    # 360 x 2 n_nodes = 360 feat_dim = 2
    # 360 x 362 n_nodes = 360 feat_dim = 362

    [train_adj_matrices, 
        train_feats, 
        train_age_labels,
        train_split_indices,
        val_adj_matrices, 
        val_feats,
        val_age_labels,
        val_split_indices, 
        test_adj_matrices, 
        test_feats,
        test_age_labels,
        test_split_indices] = get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices(threshold, num_sbjs, data_path, use_thicks_myelins, use_super_node)
    
    train_graph_data_dicts = [{'adj_train': sp.csr_matrix(train_adj), 
                                'features': train_feat,
                                'age_label': train_age_label,
                                'index': train_split_index} 
                                for train_adj, train_feat, train_age_label, train_split_index 
                                    in zip(train_adj_matrices, train_feats, train_age_labels, train_split_indices)]
    val_graph_data_dicts = [{'adj_train': sp.csr_matrix(val_adj), 
                                'features': val_feat,
                                'age_label': val_age_label,
                                'index': val_split_index} 
                                for val_adj, val_feat, val_age_label, val_split_index 
                                    in zip(val_adj_matrices, val_feats, val_age_labels, val_split_indices)]
    test_graph_data_dicts = [{'adj_train': sp.csr_matrix(test_adj), 
                                'features': test_feat,
                                'age_label': test_age_label,
                                'index': test_split_index} 
                                for test_adj, test_feat, test_age_label, test_split_index 
                                    in zip(test_adj_matrices, test_feats, test_age_labels, test_split_indices)]

    return train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts



def load_data_meg(num_sbjs : int, data_path : str, use_thicks_myelins : bool, use_super_node : bool):
    # ADJ MAT
    # 90 x 90
    # FEATURES Identity Matrix
    # Thicks Myelins is 2 x 592 x 360

    # 360 x 2 n_nodes = 360 feat_dim = 2
    # 360 x 362 n_nodes = 360 feat_dim = 362

    [train_adj_matrices, 
        train_feats, 
        train_age_labels,
        train_split_indices,
        val_adj_matrices, 
        val_feats,
        val_age_labels,
        val_split_indices, 
        test_adj_matrices, 
        test_feats,
        test_age_labels,
        test_split_indices] = get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices_meg(num_sbjs, data_path, use_thicks_myelins, use_super_node)
    
    train_graph_data_dicts = [{'adj_train': sp.csr_matrix(train_adj), 
                                'features': train_feat,
                                'age_label': train_age_label,
                                'index': train_split_index} 
                                for train_adj, train_feat, train_age_label, train_split_index 
                                    in zip(train_adj_matrices, train_feats, train_age_labels, train_split_indices)]
    val_graph_data_dicts = [{'adj_train': sp.csr_matrix(val_adj), 
                                'features': val_feat,
                                'age_label': val_age_label,
                                'index': val_split_index} 
                                for val_adj, val_feat, val_age_label, val_split_index 
                                    in zip(val_adj_matrices, val_feats, val_age_labels, val_split_indices)]
    test_graph_data_dicts = [{'adj_train': sp.csr_matrix(test_adj), 
                                'features': test_feat,
                                'age_label': test_age_label,
                                'index': test_split_index} 
                                for test_adj, test_feat, test_age_label, test_split_index 
                                    in zip(test_adj_matrices, test_feats, test_age_labels, test_split_indices)]

    return train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts



def load_data_binary_cyclic_tree_multiple(num_sbjs, data_path):
    # ADJ MAT
    # 360 x 360
    # FEATURES 592 x 2 x 360
    # Thicks Myelins is 2 x 592 x 360

    # 360 x 2 n_nodes = 360 feat_dim = 2
    # 360 x 362 n_nodes = 360 feat_dim = 362

    [train_adj_matrices, 
        train_feats, 
        train_age_labels,
        train_split_indices,
        val_adj_matrices, 
        val_feats,
        val_age_labels,
        val_split_indices, 
        test_adj_matrices, 
        test_feats,
        test_age_labels,
        test_split_indices] = get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices_for_binary_cyclic_tree_multiple(num_sbjs, data_path)
    
    train_graph_data_dicts = [{'adj_train': sp.csr_matrix(train_adj), 
                                'features': train_feat,
                                'age_label': train_age_label,
                                'index': train_split_index} 
                                for train_adj, train_feat, train_age_label, train_split_index 
                                    in zip(train_adj_matrices, train_feats, train_age_labels, train_split_indices)]
    val_graph_data_dicts = [{'adj_train': sp.csr_matrix(val_adj), 
                                'features': val_feat,
                                'age_label': val_age_label,
                                'index': val_split_index} 
                                for val_adj, val_feat, val_age_label, val_split_index 
                                    in zip(val_adj_matrices, val_feats, val_age_labels, val_split_indices)]
    test_graph_data_dicts = [{'adj_train': sp.csr_matrix(test_adj), 
                                'features': test_feat,
                                'age_label': test_age_label,
                                'index': test_split_index} 
                                for test_adj, test_feat, test_age_label, test_split_index 
                                    in zip(test_adj_matrices, test_feats, test_age_labels, test_split_indices)]

    return train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts

def get_adjacency_matrix_for_binary_cyclic_tree(depth):
    adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])

    for node_index in range(1, (2 ** depth - 1) // 2 + 1):
        adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
        adj_matrix[node_index - 1][2 * node_index] = 1.0
        adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
        adj_matrix[2 * node_index][node_index - 1] = 1.0
    for node_index in range(len(adj_matrix)):
        if (node_index + 1) % 2 == 0: 
            adj_matrix[node_index][node_index + 1] = 1.0
            adj_matrix[node_index + 1][node_index] = 1.0
    return adj_matrix

def get_adjacency_matrix_for_binary_cyclic_tree_with_super_node(depth):
    adj_matrix = np.array([[0.0 for _ in range(2 ** depth)] for _ in range(2 ** depth)])

    for node_index in range(1, (2 ** depth - 1) // 2 + 1):
        adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
        adj_matrix[node_index - 1][2 * node_index] = 1.0
        adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
        adj_matrix[2 * node_index][node_index - 1] = 1.0
    for node_index in range(len(adj_matrix) - 1):
        if (node_index + 1) % 2 == 0: 
            adj_matrix[node_index][node_index + 1] = 1.0
            adj_matrix[node_index + 1][node_index] = 1.0
    adj_matrix[-1] = [1] * len(adj_matrix)
    for i in range(len(adj_matrix)):
        adj_matrix[i][-1] = 1
    return adj_matrix


def get_dataset_split_indices(num_sbjs) -> List[List[int]]:
    
    train_split_indices, val_split_indices, test_split_indices = [], [], []
    import logging
    # logging.info("Using Only 18 - 30 Age Range")
    # logging.info("Using Only 18 - 30 Age Range")
    nth_index = 0
    nth_str = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"][nth_index]
    # num_sbjs = 70
    logging.info(f"Using Only {nth_str} {num_sbjs} Subjects")
    # TODO: Change Back to 587
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    # TODO: RESTORE BACK TO 100 % ONCE DONE TESTING SUBSET! 
    # train_num = train_num // 10
    # val_num = val_num // 10
    # test_num = test_num // 10
    
    # print("Using only one tenth of the data for training, validation and testing")
    # print("Train Num: ", train_num)
    # print("Val Num: ", val_num)
    # print("Test Num: ", test_num)

    seen = set()
    lower_bound = num_sbjs * nth_index
    upper_bound = lower_bound + num_sbjs
    # upper_bound = 587 # [490, 587) -> 97
    for split_indices, split_num in zip([train_split_indices, val_split_indices, test_split_indices],
                                    [train_num, val_num, test_num]):
        num_indices = 0
        while num_indices < split_num:
            # index = np.random.randint(0, num_sbjs) # Do not forget filtering! (subtract 5)        
            index = np.random.randint(lower_bound, upper_bound) # Do not forget filtering! (subtract 5)        
            if index in seen: continue
            split_indices.append(index)
            num_indices += 1
            seen.add(index)
    assert not set(train_split_indices).intersection(val_split_indices), "Train and Val Sets should not overlap"
    assert not set(train_split_indices).intersection(test_split_indices), "Train and Test Sets should not overlap"
    assert not set(val_split_indices).intersection(test_split_indices), "Val and Test Sets should not overlap"
    return train_split_indices, val_split_indices, test_split_indices

def get_adjacency_matrix_library_learning(graph_index : int, data_path : str) -> List[List[int]]:
    """
    Create adjacency matrix from threshold number by binarizing the float PLV matrix
    into a 1 and 0 matrix 

    """
    # First open csv file as pandas, then access pandas cell using graph_index, access activity_concept_hierarchy column
    # Then use json.loads to convert string to json, then access the adjacency matrix
    video_transcripts_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'library_learning', 'video_transcripts_with_hierarchy_1701550479.csv'))
    json_str = video_transcripts_df['activity_concept_hierarchy'][graph_index]
    # print("THIS IS JSON STR", json_str)
    try:
        json_graph = json.loads(json_str.replace("'", "\""))
    except:
        # In case dataset has a NaN Error
        print(f"Caught NaN Error, returning random adjacency matrix in graph index {graph_index}")
        return np.random.randint(0, 2, (MAX_CONCEPT_HIERARCHY_SIZE, MAX_CONCEPT_HIERARCHY_SIZE))
    # json_graph = json.load(open(os.path.join(data_path, "video_transcripts_with_hierarchy_1701550479.csv")))
    def add_nodes_edges(graph, node, parent=None):
        # Assign a unique identifier to each node using id(node)
        # node_num = np.random.randint(1000, 9999)
        node_id = f"{node['activity']}_{id(node)}"
        # node_id = f"{node['activity']}_{node_num}"
        
        # Add the current node to the graph
        if parent: graph.add_edge(parent, node_id)
        graph.add_node(node_id, primary_concept=node["primary_concept"])

        # If there are supporting concepts, add them as leaf nodes
        for supporting_concept in node.get("supporting_concepts", []):
            supporting_id = f"{supporting_concept}_{id(node)}"
            # supporting_id = f"{supporting_concept}_{node_num}"
            graph.add_node(supporting_id, primary_concept=supporting_concept)
            graph.add_edge(node_id, supporting_id)

        # If there are activities, recursively add them as sub-trees
        for activity in node.get("activities", []):
            activity_id = f"{activity['activity']}_{id(activity)}"
            # activity_num = np.random.randint(1000, 9999)
            # activity_id = f"{activity['activity']}_{activity_num}"
            graph.add_edge(node_id, activity_id)
            add_nodes_edges(graph, activity, parent=node_id)

    # Create a directed graph
    G = nx.DiGraph()
    # Add root node
    root_node_id = "lesson"
    root_primary_concept = "lesson"
    G.add_node(root_node_id, primary_concept=root_primary_concept)

    for lesson in json_graph["lesson"]:
        add_nodes_edges(G, lesson, parent=root_node_id)

    # Visualize the tree structure
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)
    # plt.show()

    # # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    # NOTE: Networkx only gives upper triangular matrix, so we add the transpose to get the full adjacency matrix
    symmetric_matrix = adjacency_matrix + adjacency_matrix.T
    # Size of the expanded matrix
    expanded_size = (MAX_CONCEPT_HIERARCHY_SIZE, MAX_CONCEPT_HIERARCHY_SIZE)

    # Create an expanded matrix filled with zeros
    expanded_matrix = np.zeros(expanded_size)

    # Copy the original matrix into the top-left corner of the expanded matrix
    expanded_matrix[:len(symmetric_matrix), :len(symmetric_matrix)] = symmetric_matrix

    return expanded_matrix

def get_adjacency_matrix_cam_can(threshold : float, graph_index : int, data_path : str, use_super_node : bool = False) -> List[List[int]]:
    """
    Create adjacency matrix from threshold number by binarizing the float PLV matrix
    into a 1 and 0 matrix 

    """
    plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
    plv_tensor = remove_self_loops(plv_tensor)
    plv_matrix = plv_tensor[graph_index].copy()
    import logging
    # logging.info(f"USING HIGHEST HIDDEN LAYER YET 1024")
    logging.info(f"Using PLV Threshold : {threshold}")
    plv_matrix [plv_matrix < threshold] = 0
    plv_matrix [plv_matrix >= threshold] = 1
    
    # add_probability = 0.75    
    # # logging.info(f"Hierarchy Test: With some probability p = {add_probability}, Add edges from ROI vectors that correspond to ACMP/DP")
    # logging.info(f"Hierarchy Test: With some probability p = {add_probability}, Remove edges from ROI vectors that correspond to MT/LT")
    # logging.info("PREVIOUS ADJACENCY MATRIX: \n {}".format(plv_matrix))
    # ACMP_SUBNET_ID = 19
    # DP_SUBNET_ID = 22
    # MT_ID = 13
    # LT_ID = 14
    # from visualization import get_hcp_atlas_df
    # hcp_atlas_df = get_hcp_atlas_df()
    # cortex_ids = hcp_atlas_df['Cortex_ID']
    # for i in range(NUM_ROIS):
    #     if cortex_ids[i] == ACMP_SUBNET_ID:
    #         # NOTE : Binarized matrix might already have 1's at [i][j]
    #         for j in range(NUM_ROIS):
    #             # if plv_matrix[i][j] == 1:
    #             #     if np.random.uniform() < add_probability:
    #             #         plv_matrix[i][j] = 0
    #             #         plv_matrix[j][i] = 0    
    #             if not plv_matrix[i][j]:
    #                 if np.random.uniform() < add_probability:
    #                     plv_matrix[i][j] = 1
    #                     plv_matrix[j][i] = 1
    # logging.info("ADJACENCY MATRIX AFTER ADDING: \n {}".format(plv_matrix))
    if use_super_node:
        logging.info("Using Super Node")
        plv_matrix = np.insert(plv_matrix, 0, 1, axis = 0)
        plv_matrix = np.insert(plv_matrix, 0, 1, axis = 1)
        plv_matrix[-1] = [1] * len(plv_matrix)
        for i in range(len(plv_matrix)):
            plv_matrix[i][-1] = 1
        plv_matrix[-1][-1] = 0

    return plv_matrix


def get_adjacency_matrix_meg(graph_index : int, data_path : str, use_super_node : bool = False) -> List[List[int]]:
    """
    Create adjacency matrix from threshold number by binarizing the float PLV matrix
    into a 1 and 0 matrix 

    """
    plv_tensor = np.load(os.path.join(data_path, "MEG.ROIs.npy"))
    # plv_tensor = remove_self_loops(plv_tensor)
    plv_matrix = plv_tensor[graph_index][MEG_BAND_ALPHA_INDEX][MEG_PLV_METRIC_INDEX].copy()

    plv_matrix [plv_matrix < THRESHOLD_MEG] = 0
    plv_matrix [plv_matrix >= THRESHOLD_MEG] = 1
    
    if use_super_node:
        plv_matrix = np.insert(plv_matrix, 0, 1, axis = 0)
        plv_matrix = np.insert(plv_matrix, 0, 1, axis = 1)
        plv_matrix[-1] = [1] * len(plv_matrix)
        for i in range(len(plv_matrix)):
            plv_matrix[i][-1] = 1
        plv_matrix[-1][-1] = 0

    return plv_matrix

def get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices_for_binary_cyclic_tree_multiple(num_sbjs : int, data_path):
    
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    assert train_num + val_num + test_num == num_sbjs, "Must add to total number of subjects"
    
    train_split_indices, val_split_indices, test_split_indices = [0, 100, 200], [250, 450], [125, 525]

    train_split_adj_matrices = [get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) for i in train_split_indices]
    val_split_adj_matrices = [get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) for i in val_split_indices]
    test_split_adj_matrices = [get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH) for i in test_split_indices]
    adj_mat = train_split_adj_matrices[0]
    use_noise = False
    if use_noise:
        train_feats = [np.eye(len(train_split_adj_matrices[0])) + np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat))) for i in train_split_indices]
        val_feats = [np.eye(len(train_split_adj_matrices[0])) + np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat))) for i in val_split_indices]
        test_feats = [np.eye(len(train_split_adj_matrices[0])) + np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat))) for i in test_split_indices]
    else:
        train_feats = [np.eye(len(train_split_adj_matrices[0])) for i in train_split_indices]
        val_feats = [np.eye(len(train_split_adj_matrices[0])) for i in val_split_indices]
        test_feats = [np.eye(len(train_split_adj_matrices[0])) for i in test_split_indices]

    # if use_noise:
    #     train_feats = [np.flipud(np.eye(len(train_split_adj_matrices[0]))) + np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat))) for i in train_split_indices]
    #     val_feats = [np.flipud(np.eye(len(train_split_adj_matrices[0]))) + np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat))) for i in val_split_indices]
    #     test_feats = [np.flipud(np.eye(len(train_split_adj_matrices[0]))) + np.random.normal(0, 0.01, (len(adj_mat), len(adj_mat))) for i in test_split_indices]
    # else:
    #     train_feats = [np.flipud(np.eye(len(train_split_adj_matrices[0]))) for i in train_split_indices]
    #     val_feats = [np.flipud(np.eye(len(train_split_adj_matrices[0]))) for i in val_split_indices]
    #     test_feats = [np.flipud(np.eye(len(train_split_adj_matrices[0]))) for i in test_split_indices]

    age_labels = np.load(os.path.join(data_path, "age_labels_592_sbj_filtered.npy"))
    train_age_labels = [age_labels[i] for i in train_split_indices]
    val_age_labels = [age_labels[i] for i in val_split_indices]
    test_age_labels = [age_labels[i] for i in test_split_indices]

    return [train_split_adj_matrices, 
            train_feats, 
            train_age_labels,
            train_split_indices, 
            val_split_adj_matrices, 
            val_feats, 
            val_age_labels,
            val_split_indices,
            test_split_adj_matrices, 
            test_feats, 
            test_age_labels,
            test_split_indices]


def get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices(threshold : float, num_sbjs : int, data_path : str, use_thicks_myelins : bool, use_super_node : bool):
    
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    assert train_num + val_num + test_num == num_sbjs, "Must add to total number of subjects"
    train_split_indices, val_split_indices, test_split_indices = get_dataset_split_indices(num_sbjs)
    # NOTE: Using subset of data for now to test and fix clumping issue, will use full dataset once clumping issue is fixed
    # UPDATE: Seem to have fixed issue! Now using full dataset
    # train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]
    # import logging
    # logging.info("Using 5 Graphs")
    # train_split_indices, val_split_indices, test_split_indices = [100, 100, 100, 100, 100], [100, 100, 100], [100, 100]
    # train_split_indices, val_split_indices, test_split_indices = [100, 100, 100], [100], [100]
    # train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]
    

    # TODO: Remove this once hierarchy testing is complete!
    # Must ensure same adjacency matrix with additional edges fed into model for each split to test hierarchy changes
    # logging.info("Using same adjacency matrix to test Hierarchy Changes!")
    # adjacency_matrix = get_adjacency_matrix_cam_can(100, data_path, use_super_node)
    # train_split_adj_matrices = [adjacency_matrix for _ in train_split_indices]
    # val_split_adj_matrices = [adjacency_matrix for _ in val_split_indices]
    # test_split_adj_matrices = [adjacency_matrix for _ in test_split_indices]

    train_split_adj_matrices = [get_adjacency_matrix_cam_can(threshold, i, data_path, use_super_node) for i in train_split_indices]
    val_split_adj_matrices = [get_adjacency_matrix_cam_can(threshold, i, data_path, use_super_node) for i in val_split_indices]
    test_split_adj_matrices = [get_adjacency_matrix_cam_can(threshold, i, data_path, use_super_node) for i in test_split_indices]
    
    train_feats = [get_feature_matrix(i, data_path, use_thicks_myelins, use_super_node) for i in train_split_indices]
    val_feats = [get_feature_matrix(i, data_path, use_thicks_myelins, use_super_node) for i in val_split_indices]
    test_feats = [get_feature_matrix(i, data_path, use_thicks_myelins, use_super_node) for i in test_split_indices]

    age_labels = np.load(os.path.join(data_path, "age_labels_592_sbj_filtered.npy"))
    train_age_labels = [age_labels[i] for i in train_split_indices]
    val_age_labels = [age_labels[i] for i in val_split_indices]
    test_age_labels = [age_labels[i] for i in test_split_indices]

    return [train_split_adj_matrices, 
            train_feats, 
            train_age_labels,
            train_split_indices, 
            val_split_adj_matrices, 
            val_feats, 
            val_age_labels,
            val_split_indices,
            test_split_adj_matrices, 
            test_feats, 
            test_age_labels,
            test_split_indices]

def get_adj_mat_dataset_splits_with_features_and_age_labels_and_indices_meg(num_sbjs : int, data_path : str, use_thicks_myelins : bool, use_super_node : bool):
    
    train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
    test_num = num_sbjs - train_num - val_num 
    assert train_num + val_num + test_num == num_sbjs, "Must add to total number of subjects"
    train_split_indices, val_split_indices, test_split_indices = get_dataset_split_indices(num_sbjs)
    # NOTE: Using subset of data for now to test and fix clumping issue, will use full dataset once clumping issue is fixed
    # UPDATE: Seem to have fixed issue! Now using full dataset
    # train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]
    # import logging
    # logging.info("Using 5 Graphs")
    # train_split_indices, val_split_indices, test_split_indices = [100, 100, 100, 100, 100], [100, 100, 100], [100, 100]
    # train_split_indices, val_split_indices, test_split_indices = [100, 100, 100], [100], [100]
    # train_split_indices, val_split_indices, test_split_indices = [0, 100, 200, 300, 400, 500], [50, 250, 450], [125, 525]
    
    # TODO: Remove this once hierarchy testing is complete!
    # Must ensure same adjacency matrix with additional edges fed into model for each split to test hierarchy changes
    # logging.info("Using same adjacency matrix to test Hierarchy Changes!")
    # adjacency_matrix = get_adjacency_matrix_cam_can(100, data_path, use_super_node)
    # train_split_adj_matrices = [adjacency_matrix for _ in train_split_indices]
    # val_split_adj_matrices = [adjacency_matrix for _ in val_split_indices]
    # test_split_adj_matrices = [adjacency_matrix for _ in test_split_indices]

    train_split_adj_matrices = [get_adjacency_matrix_meg(i, data_path, use_super_node) for i in train_split_indices]
    val_split_adj_matrices = [get_adjacency_matrix_meg(i, data_path, use_super_node) for i in val_split_indices]
    test_split_adj_matrices = [get_adjacency_matrix_meg(i, data_path, use_super_node) for i in test_split_indices]
    
    train_feats = [np.eye(NUM_MEG_COLE_ROIS) for _ in train_split_indices]
    val_feats = [np.eye(NUM_MEG_COLE_ROIS) for _ in val_split_indices]
    test_feats = [np.eye(NUM_MEG_COLE_ROIS) for _ in test_split_indices]

    age_labels = np.load(os.path.join(data_path, "MEG_age_labels_180.npy"))
    train_age_labels = [age_labels[i] for i in train_split_indices]
    val_age_labels = [age_labels[i] for i in val_split_indices]
    test_age_labels = [age_labels[i] for i in test_split_indices]

    return [train_split_adj_matrices, 
            train_feats, 
            train_age_labels,
            train_split_indices, 
            val_split_adj_matrices, 
            val_feats, 
            val_age_labels,
            val_split_indices,
            test_split_adj_matrices, 
            test_feats, 
            test_age_labels,
            test_split_indices]


def remove_self_loops(plv_tensor):
    for sbj_index in range(len(plv_tensor)):
        for i in range(NUM_ROIS):
            plv_tensor[sbj_index][i][i] = 0
    return plv_tensor

def get_feature_matrix(graph_index : int, data_path : str, use_thicks_myelins : bool = True, use_super_node : bool = False):
    """
    Feature Matrix for specific subject if 360 x 362 since PLV_vectors are 360
    [CT_0   Myelin_0      PLV_vector_0.T      ]
    [CT_1   Myelin_1      PLV_vector_1.T      ]
    ...
    [CT_359 Myelin_359    PLV_vector_359.T    ]

    """
    plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))
    
    plv_tensor = remove_self_loops(plv_tensor)
    thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
    # Thicks Myelins is 2 x 592 x 360
    # TODO: Make PLV Tensor and Thickness + Myelins Tensor global to avoid 
    # constant loading
    
    plv_features = plv_tensor[graph_index].copy()
    
    thick_features = thicks_myelins_tensor[THICK_INDEX][graph_index]
    thick_features = np.reshape(thick_features, (len(thick_features), 1))

    myelin_features = thicks_myelins_tensor[MYELIN_INDEX][graph_index] 
    myelin_features = np.reshape(myelin_features, (len(myelin_features), 1))

    use_min_max_scaler = True
    use_z_score_normalization = False
    if use_min_max_scaler:
        # TODO: PLOT THE MIN MAX NORMALIZED FEATURES FOR EACH SUBJECT
        thick_features = min_max_normalize(thick_features)
        myelin_features = min_max_normalize(myelin_features)    
        plv_features = min_max_normalize(plv_features)
    if use_z_score_normalization:
        thick_features = z_score_normalize(thick_features)
        myelin_features = z_score_normalize(myelin_features)
    # TODO: Make these more notorious! Can lead to confusion with results 
    # from not remembering that these are activated and therefore not using 
    # all features inadvertently!
    import logging
    logging.getLogger().setLevel(logging.INFO)
    
    # logging.info("Using Myelination + PLV Matrix")
    # feature_matrix = np.hstack((myelin_features, plv_features))
    use_only_thicks = True
    use_only_myelins = False
    if use_only_thicks:
        logging.info("Using CT + Identity Feature Matrix")
        feature_matrix = np.hstack((thick_features, np.eye(NUM_ROIS)))
    elif use_only_myelins:
        logging.info("Using Myelination + Identity Feature Matrix")
        feature_matrix = np.hstack((myelin_features, np.eye(NUM_ROIS)))
    elif use_thicks_myelins:
        logging.info("Using Myelination + PLV + CT Feature Matrix")
        feature_matrix = np.hstack((myelin_features, plv_features, thick_features))
        # logging.info("Using CT + Myelination + PLV  Feature Matrix")
        # feature_matrix = np.hstack((thick_features, myelin_features, plv_features))
    else:
        logging.info("Using PLV Feature Matrix only")
        # logging.info("Added Identity Matrix to Feature Matrix!")
        # feature_matrix = np.hstack((plv_features, np.eye(NUM_ROIS)))
        feature_matrix = plv_features
    # TODO: Change back to correct feature matrix
    # logging.info("Using CT + Myelination + Identity Feature Matrix")
    # feature_matrix = np.hstack((thick_features, myelin_features, np.eye(NUM_ROIS)))
    logging.info(f"Use Super Node : {use_super_node}")
    if use_super_node:
        # feature_matrix = np.eye(NUM_ROIS + 1)
        num_rows, num_cols = feature_matrix.shape

        # Create a new matrix with an extra row and an extra column
        new_matrix = np.zeros((num_rows + 1, num_cols + 1), dtype=feature_matrix.dtype)

        # Copy the elements from the original matrix to the new matrix
        new_matrix[ : num_rows, : num_cols] = feature_matrix

        # Set the bottom-right corner to 1
        new_matrix[num_rows, num_cols] = 1
        feature_matrix = new_matrix
    # use_adj_matrix_powers = False
    # def get_adjacency_matrix_powers():
    #     adj_mat = get_adjacency_matrix_cam_can(graph_index, data_path)
    #     shrink_factor = 0.4
    #     return np.linalg.inv(np.eye(NUM_ROIS) - shrink_factor * adj_mat)

    # if use_adj_matrix_powers:
    #     logging.info("Using Adjacency Matrix Powers")

    #     adj_mat_pow = get_adjacency_matrix_powers()
    #     logging.info("Adj Mat Powers: {}".format(adj_mat_pow))
    #     feature_matrix = np.hstack((thick_features, myelin_features, adj_mat_pow))
        
    # else:
    #     feature_matrix = np.hstack((thick_features, myelin_features, np.eye(NUM_ROIS)))

    # TODO: Check how CT + Identity changes if we don't use random noise
    use_noise = False
    if use_noise:
        feature_matrix += np.random.normal(0, 0.01, size=feature_matrix.shape)
    # if use_min_max_scaler:
    #     feature_matrix = min_max_normalize(feature_matrix)
    logging.info(f"Feature Matrix with No Noise: {feature_matrix}")
    return feature_matrix

def get_thick_features(graph_index : int, data_path : str):
    thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy"))
    thick_features = thicks_myelins_tensor[THICK_INDEX][graph_index]
    return thick_features

def get_myelin_features(graph_index : int, data_path : str):
    thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy"))
    myelin_features = thicks_myelins_tensor[MYELIN_INDEX][graph_index] 
    return myelin_features
# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

