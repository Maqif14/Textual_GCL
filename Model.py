import torch
import torch.nn as nn
import torch.nn.functional as F
from util.sampler import next_batch_pairwise
from util.torch_interface import TorchGraphInterface
from util.loss import bpr_loss, l2_reg_loss, InfoNCE
from util.ui_graph import Interaction
from util.algorithm import find_k_largest
from util.evaluation import ranking_evaluation
from transformers import AutoModel
from tqdm import tqdm
from util.sampler import string_to_tensor



class Model(Interaction):
    def __init__(self, training_data, test_data, device):
        self.data = Interaction(training_data, test_data)
        self.ranking = [10,20]
        self.cl_rate = 0.2
        self.eps = 0.2
        self.temp = 0.15
        self.n_layers = 2
        self.layer_cl = 1
        self.batch_size = 24
        self.emb_size = 24
        self.lRate = 1e-2 
        self.maxEpoch = 5
        self.reg = 0.0001
        self.model = Model_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl, device)

        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

        self.result = []
        self.recOutput = []
        self.bestPerformance = []

    def train(self, device):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):

            for n, batch in tqdm(enumerate(next_batch_pairwise(self.data, self.batch_size))):
                paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg = batch
                paper_idx, pos_idx, neg_idx = paper_idx.to(device), pos_idx.to(device), neg_idx.to(device)

                ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg = ip_id_p.to(device), attn_msk_p.to(device), ip_id_pos.to(device), attn_mask_pos.to(device), ip_id_neg.to(device), attn_mask_neg.to(device)
                
                paper_emb, pos_dataset_emb, neg_dataset_emb, rec_paper_emb, rec_dataset_emb, cl_paper_emb, cl_dataset_emb  = model(paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device, perturbed=True)
                
                rec_loss = bpr_loss(paper_emb, pos_dataset_emb, neg_dataset_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([paper_idx, pos_idx], rec_paper_emb, cl_paper_emb, rec_dataset_emb, cl_dataset_emb, device)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, paper_emb, pos_dataset_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
            with torch.no_grad():
                self.paper_emb, self.dataset_emb = self.model(paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device)
            self.fast_evaluation(epoch, paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device)
        self.paper_emb, self.dataset_emb = self.best_paper_emb, self.best_dataset_emb 
        torch.save(self.paper_emb, './best_paper_emb.pt')
        torch.save(self.dataset_emb, '.best_dataset_emb.pt')     

    def cal_cl_loss(self, idx, paper_view1, paper_view2, dataset_view1, dataset_view2, device):
        p_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(device)
        d_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(device)
        paper_cl_loss = InfoNCE(paper_view1[p_idx], paper_view2[p_idx], self.temp)
        dataset_cl_loss = InfoNCE(dataset_view1[d_idx], dataset_view2[d_idx], self.temp)
        return paper_cl_loss + dataset_cl_loss

    def save(self, paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device):
        with torch.no_grad():
            self.best_paper_emb, self.best_dataset_emb = self.model.forward(paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device)

    def predict(self, u, input_ids_p, attention_mask_p):
        u = self.data.get_paper_id(u)
        
        # Forward pass with textual information
        with torch.no_grad():
            paper_emb = self.model.text_encoder(input_ids = input_ids_p, attention_mask=attention_mask_p)[0][:, 0, :]
        
        paper_emb = self.model.lin_paper_test(paper_emb)
        paper_emb = paper_emb.squeeze()

        # Compute score
        score = torch.matmul(paper_emb, self.dataset_emb.transpose(0, 1).cuda())
        
        # return score.cpu().numpy()
        return score.detach().cpu().numpy()
        
    def test(self, device, load_saved=False):

        if load_saved:
            # Load the saved embeddings
            self.paper_emb = torch.load('./best_paper_emb.pt').to(device)
            self.dataset_emb = torch.load('./best_dataset_emb.pt').to(device)
            self.model.to(device)

        rec_list = {}
        paper_count = len(self.data.test_set)
        paper_test = list(self.data.test_set.items())


        for i, paper in tqdm(paper_test):
            dataset_num = list(paper.keys())[0]
            paper_inputs_ids = string_to_tensor(paper[dataset_num]['input_ids_p'])
            attention_mask_p = string_to_tensor(paper[dataset_num]['attention_mask_p'])

            paper_input_ids = torch.tensor(paper_inputs_ids).unsqueeze(0).to(device) 
            paper_attn_mask = torch.tensor(attention_mask_p).unsqueeze(0).to(device)

            candidates = self.predict(i, paper_input_ids, paper_attn_mask)
           
            rated_list, _ = self.data.paper_rated(i)
            for dataset in rated_list:
                candidates[self.data.dataset[dataset]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            dataset_names = [self.data.id2dataset[did] for did in ids]
            rec_list[i] = list(zip(dataset_names, scores)) # tgok balik kat sini

        return rec_list
    
    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for paper in self.data.test_set:
            line = paper + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[paper] else ''}"
                for item in rec_list[paper]
            )
            line += '\n'
            self.recOutput.append(line)
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        print(f'The result of XSimGCL:\n{"".join(self.result)}')

    def fast_evaluation(self, epoch, paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device):
        print('Evaluating the model...')
        rec_list = self.test(device)
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save(paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device)
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save(paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device)

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Dataset Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure
    
class Model_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl, device):
        super(Model_Encoder, self).__init__()
        self.data = data
        self.text_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

        self.lin_all_paper_embedding = nn.Linear(768, self.emb_size)
        self.lin_all_dataset_embedding = nn.Linear(768, self.emb_size)

        self.shape_size = 792

        self.paper_pre = nn.Linear(self.shape_size, self.shape_size)
        self.paper_post = nn.Linear(self.shape_size, 64)

        self.paper_pre = nn.Linear(self.shape_size, self.shape_size)
        self.paper_post = nn.Linear(self.shape_size, 64)

        self.paper_pre = nn.Linear(self.shape_size, self.shape_size)
        self.paper_post = nn.Linear(self.shape_size, 64)

        self.lin_paper_test = nn.Linear(768, self.emb_size)


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'paper_emb': nn.Parameter(initializer(torch.empty(self.data.paper_num, self.emb_size))),
            'dataset_emb': nn.Parameter(initializer(torch.empty(self.data.dataset_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, paper_idx, pos_idx, neg_idx, ip_id_p, attn_msk_p, ip_id_pos, attn_mask_pos, ip_id_neg, attn_mask_neg, device, perturbed=False):
        
        ego_embeddings = torch.cat([self.embedding_dict['paper_emb'], self.embedding_dict['dataset_emb']], 0)
        
        sparse_norm_adj = self.sparse_norm_adj.to(ego_embeddings.device)
        
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(device)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        graph_embeddings = torch.stack(all_embeddings, dim=1)
        graph_embeddings = torch.mean(graph_embeddings, dim=1)
        paper_graph_all_embeddings, dataset_graph_all_embeddings = torch.split(graph_embeddings, [self.data.paper_num, self.data.dataset_num])
        paper_graph_all_embeddings_cl, dataset_graph_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.paper_num, self.data.dataset_num])
        
        # Textual Embedding

        paper_emb_textual = self.text_encoder(input_ids = ip_id_p, attention_mask=attn_msk_p)[0][:, 0, :]
        dataset_pos_emb_textual = self.text_encoder(input_ids = ip_id_pos, attention_mask=attn_mask_pos)[0][:, 0, :]
        dataset_neg_emb_textual = self.text_encoder(input_ids = ip_id_neg, attention_mask=attn_mask_neg)[0][:, 0, :]
        paper_batch_emb, data_pos_emb, data_neg_emb = paper_graph_all_embeddings[paper_idx], dataset_graph_all_embeddings[pos_idx], dataset_graph_all_embeddings[neg_idx]

        all_paper_emb = torch.matmul(paper_graph_all_embeddings, paper_emb_textual)
        all_paper_emb = self.lin_all_paper_embedding(all_paper_emb)

        
        all_dataset_emb = torch.matmul(dataset_graph_all_embeddings, dataset_pos_emb_textual)
        all_dataset_emb = self.lin_all_dataset_embedding(all_dataset_emb)

        all_paper_emb_cl = torch.matmul(paper_graph_all_embeddings_cl, paper_emb_textual)
        all_paper_emb_cl = self.lin_all_paper_embedding(all_paper_emb_cl)

        all_dataset_emb_cl = torch.matmul(dataset_graph_all_embeddings_cl, dataset_pos_emb_textual)
        all_dataset_emb_cl = self.lin_all_dataset_embedding(all_dataset_emb_cl)

        paper_batch_emb = torch.cat((paper_batch_emb, paper_emb_textual), dim=1)
        pos_dataset_batch_emb = torch.cat((data_pos_emb, dataset_pos_emb_textual), dim=1)
        neg_dataset_batch_emb = torch.cat((data_neg_emb, dataset_neg_emb_textual), dim=1)
        

        paper_batch_emb = self.paper_pre(paper_batch_emb)
        paper_batch_emb = self.paper_post(paper_batch_emb)

        pos_dataset_batch_emb = self.paper_pre(pos_dataset_batch_emb)
        pos_dataset_batch_emb = self.paper_post(pos_dataset_batch_emb)

        neg_dataset_batch_emb = self.paper_pre(neg_dataset_batch_emb)
        neg_dataset_batch_emb = self.paper_post(neg_dataset_batch_emb)


        if perturbed:
            return paper_batch_emb, pos_dataset_batch_emb, neg_dataset_batch_emb, all_paper_emb, all_dataset_emb, all_paper_emb_cl, all_dataset_emb_cl
        return all_paper_emb, all_dataset_emb
    

