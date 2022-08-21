import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np
import dgl
from logzero import logger


__all__ = ['Network']
device = torch.device('cuda:0')

class Network(nn.Module):
    def __init__(self, *, labels_num, input_size, hidden_size, num_gcn=0, dropout=0.5, pretrain_embedding2, interpro_size, **kwargs):
        super(Network, self).__init__()
        logger.info(F'labels_num={labels_num}, input size={input_size}, hidden_size={hidden_size}')
        self.labels_num = labels_num
        self.num_gcn = num_gcn
        self.protein_embedding = pretrain_embedding2
        
        # input layer
        self.input = nn.EmbeddingBag(interpro_size, 1024, mode='sum', include_last_offset=True, scale_grad_by_freq=True)
        self.esmlinear = nn.Linear(1280, 1024)
        self.input_bias1 = nn.Parameter(torch.zeros(1024))
        self.input_bias2 = nn.Parameter(torch.zeros(1024)) 
        # GraphSAGE layer
        self.update = nn.ModuleList(NodeUpdate(2048, 1024, dropout) for _ in range(num_gcn))
        # fusion layer
        self.w = nn.Parameter(torch.ones(2))
        # classification layer
        self.output = nn.Linear(2304, self.labels_num)  

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()


    def forward(self, nf: dgl.NodeFlow, inputs, inputs_i, target_id):
        nf.copy_from_parent()
        x1 = Fun.relu(self.esmlinear(torch.from_numpy(np.array(self.protein_embedding[inputs])).cuda()) + self.input_bias1)
        x2 = Fun.relu(self.input(*inputs_i) + self.input_bias2)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        outputs = w1 * x1 + w2 * x2
        nf.layers[0].data['h'] = outputs

        for i, update in enumerate(self.update):
            nf.block_compute(i,
                                dgl.function.u_mul_e('h', 'self', out='m_res'),
                                dgl.function.sum(msg='m_res', out='res')) 
            nf.block_compute(i,
                             dgl.function.u_mul_e('h', 'ppi', out='ppi_m_out'), 
                             dgl.function.sum(msg='ppi_m_out', out='ppi_out'), update)
        


        last = torch.cat((nf.layers[-1].data['h'], torch.from_numpy(np.array(self.protein_embedding[target_id])).cuda()), 1)
        return self.output(last)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.esmlinear.weight)
        for update in self.update:
            update.reset_parameters()
        nn.init.xavier_uniform_(self.output.weight)


class NodeUpdate(nn.Module):

    def __init__(self, in_f, out_f, dropout):
        super(NodeUpdate, self).__init__()
        self.ppi_linear = nn.Linear(in_f, out_f)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node):
        outputs = self.dropout(Fun.relu(self.ppi_linear(torch.cat((node.data['ppi_out'] , node.data['res']) ,1))))
        return {'h': outputs}

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ppi_linear.weight)