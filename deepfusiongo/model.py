import torch
import torch.nn as nn
import numpy as np
import dgl

from pathlib import Path
from tqdm import tqdm
from logzero import logger

from deepfusiongo.architecture import Network
from deepfusiongo.evaluation import fmax, aupr

__all__ = ['Model']


class Model(object):
    def __init__(self, *, model_path: Path, dgl_graph, pretrain_embedding, interpro, **kwargs):
        self.dgl_graph, self.pretrain_embedding, self.interpro, self.batch_size = dgl_graph, pretrain_embedding, interpro, None
        
        self.model = Network(**kwargs)
        self.model.cuda()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())


    # train

    def train_step(self, train_pid, train_go, **kwargs):
        self.model.train()

        batch_x_interpro = self.interpro[train_pid.layer_parent_nid(0).numpy()] 
        batch_x_pretrain_emb = train_pid.layer_parent_nid(0).numpy()
        target_id = train_pid.layer_parent_nid(2).numpy()
        result = self.model(train_pid, batch_x_pretrain_emb,(torch.from_numpy(batch_x_interpro.indices).cuda().long(),
                                   torch.from_numpy(batch_x_interpro.indptr).cuda().long(),
                                   torch.from_numpy(batch_x_interpro.data).cuda().float()), target_id)

        loss = self.loss(result, train_go.cuda())
        loss.backward()
        self.optimizer.step(closure=None)
        self.optimizer.zero_grad()
        return loss.item()

    def train(self, train_data, valid_data, epochs_num, batch_size, **kwargs):
        self.batch_size = batch_size
        (train_pid, train_go), (valid_ppi, valid_y) = train_data, valid_data
        ppi_train_id = np.full(self.pretrain_embedding.shape[0], -1, dtype=np.int)
        ppi_train_id[train_pid] = np.arange(train_pid.shape[0])

        best_fmax = 0.0
        for epoch_idx in range(epochs_num):
            train_loss = 0.0
            for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, batch_size, 
                                                                        self.dgl_graph.number_of_nodes(), 
                                                                        num_hops=self.model.num_gcn,
                                                                        seed_nodes=train_pid,
                                                                        prefetch=True, shuffle=True),
                           desc=F'Epoch {epoch_idx}', leave=False, dynamic_ncols=True,
                           total=(len(train_pid) + batch_size - 1) // batch_size):
                batch_y = train_go[ppi_train_id[nf.layer_parent_nid(-1).numpy()]].toarray()
                train_loss += self.train_step(nf, torch.from_numpy(batch_y)) 
            best_fmax = self.valid(valid_ppi, valid_y, epoch_idx, train_loss / len(train_pid), best_fmax)

    # valid

    def valid(self, valid_loader, targets, epoch_idx, train_loss, best_fmax):
        result = self.predict(valid_loader, valid=True)
        (fmax_, t_), aupr_ = fmax(targets, result), aupr(targets.toarray().flatten(), result.flatten())
        logger.info(F'Epoch {epoch_idx}: Loss: {train_loss:.5f} '
                    F'Fmax: {fmax_:.3f} {t_:.2f} AUPR: {aupr_:.3f}')
        if fmax_ > best_fmax:
            best_fmax = fmax_
            self.save_model()
        return best_fmax

    @torch.no_grad()
    def predict_step(self, data_x):
        self.model.eval()

        batch_x_interpro = self.interpro[data_x.layer_parent_nid(0).numpy()] 
        batch_x_pretrain_emb = data_x.layer_parent_nid(0).numpy()
        target_id = data_x.layer_parent_nid(2).numpy()
        result = self.model(data_x, batch_x_pretrain_emb,(torch.from_numpy(batch_x_interpro.indices).cuda().long(),
                                   torch.from_numpy(batch_x_interpro.indptr).cuda().long(),
                                   torch.from_numpy(batch_x_interpro.data).cuda().float()), target_id)

        predict_result = torch.sigmoid(result).cpu().numpy()
        return predict_result

    def predict(self, test_ppi, batch_size=None, valid=False, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        if not valid:
            self.load_model()
        test_ppi_u = np.unique(test_ppi)
        test_map = {x: i for i, x in enumerate(test_ppi_u)}
        test_ppi = np.asarray([test_map[x] for x in test_ppi])
        result = np.vstack([self.predict_step(nf)
                            for nf in dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, batch_size,
                                                                                   self.dgl_graph.number_of_nodes(),
                                                                                   num_hops=self.model.num_gcn,
                                                                                   seed_nodes=test_ppi_u,
                                                                                   prefetch=True)])
        return result[test_ppi]


    # save and load 

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))