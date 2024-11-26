import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import sys

class Interaction:
    def __init__(self, training_data, test_data):
        
        self.training_data = training_data
        self.test_data = test_data

        self.paper = {}
        self.dataset = {}
        self.id2paper = {}
        self.id2dataset = {}

        self.training_set_u = defaultdict(dict) 
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_dataset = set()

        self.__generate_set()
        self.paper_num = len(self.training_set_u)
        self.dataset_num = len(self.training_set_i)

        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()


    def __generate_set(self):
        for paper, dataset, rating, id_p, am_p, id_d, am_d in self.training_data:
            if paper not in self.paper:
                paper_id = len(self.paper)
                self.paper[paper] = paper_id
                self.id2paper[paper_id] = paper
            
            if dataset not in self.dataset:
                dataset_id = len(self.dataset)
                self.dataset[dataset] = dataset_id
                self.id2dataset[dataset_id] = dataset
            
            if paper not in self.training_set_u:
                self.training_set_u[paper] = {}  
            
            self.training_set_u[paper][dataset] = {
                'rating': rating,
                'input_ids_p': id_p,
                'attention_mask_p': am_p
            }

            if dataset not in self.training_set_i:
                self.training_set_i[dataset] = {}  
            
            self.training_set_i[dataset][paper] = {
                'rating': rating,
                'input_ids_d': id_d,
                'attention_mask_d': am_d
            }
        
        for paper, dataset, rating, id_p, am_p, id_d, am_d in self.test_data:
            if paper in self.paper and dataset in self.dataset:
                if paper not in self.test_set:
                    self.test_set[paper] = {}
                
                self.test_set[paper][dataset] = {
                    'rating': rating,
                    'input_ids_p': id_p,
                    'attention_mask_p': am_p
                }
                self.test_set_dataset.add(dataset)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.paper_num + self.dataset_num
        paper_np = np.array([self.paper[pair[0]] for pair in self.training_data])
        dataset_np = np.array([self.dataset[pair[1]] for pair in self.training_data]) + self.paper_num
        ratings = np.ones_like(paper_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (paper_np, dataset_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat
    
    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat
    
    def convert_to_laplacian_mat(self, adj_mat):
        paper_np_keep, dataset_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (paper_np_keep, dataset_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)
    
    def __create_sparse_interaction_matrix(self):
        row = np.array([self.paper[pair[0]] for pair in self.training_data])
        col = np.array([self.dataset[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.paper_num, self.dataset_num), dtype=np.float32)

    def paper_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def dataset_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())
    
    def get_paper_id(self, u):
        return self.paper.get(u)

    def get_dataset_id(self, i):
        return self.dataset.get(i)