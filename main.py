from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import scipy.sparse as ssp
import dgl.data

import click
from ruamel.yaml import YAML
from logzero import logger
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

from deepfusiongo.model import Model
from deepfusiongo.psiblast_utils import blast

torch.cuda.set_device(0)

@click.command()
@click.option('-d', '--dataset-yaml', type=click.Path(exists=True))
@click.option('--model-id', type=click.INT, default=None)
def main(dataset_yaml, model_id):
    yaml = YAML(typ='safe') 
    dataset_yaml = yaml.load(Path(dataset_yaml))  # list
    model_yaml = yaml.load(Path('settings/model.yaml'))
    # blastdb
    net_blastdb = dataset_yaml['network']['blastdb']   

    # which dataset: mf/bp/cc
    data_name = dataset_yaml['dataset']
    logger.info(F'Dataset: {data_name}')

    # map of ppi
    with open(dataset_yaml['network']['pid_list']) as fp:
        ppi_pid_list = [line.split()[0] for line in fp]
    ppi_pid_map = {pid: i for i, pid in enumerate(ppi_pid_list)}  # map，key：pid，value：i

    #dgl graph
    dgl_graph = dgl.data.utils.load_graphs(dataset_yaml['network']['dgl'])[0][0] 
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_:=np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0 # self to self
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = self_loop.float().cuda()

    # fetures: interpro and pre-train embedding
    interpro = ssp.load_npz(dataset_yaml['network']['feature'])  # csr
    pretrain_embedding = np.load(dataset_yaml['network']['emb'])

    # training set and validation set
    train_pid_list, train_go = get_pid_and_go(**dataset_yaml['train'])
    valid_pid_list, valid_go = get_pid_and_go(**dataset_yaml['valid'])
    mlb = get_go_mlb(Path(dataset_yaml['mlb']), train_go)
    train_go, valid_go = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)
    *_, train_pid, train_go = get_ppi_idx(False, train_pid_list, train_go, ppi_pid_map) 
    *_, valid_pid, valid_go = get_ppi_idx(True, valid_pid_list, valid_go, ppi_pid_map, 
                                                dataset_yaml['valid']['fasta_file'], net_blastdb,
                                                Path(dataset_yaml['results'])/F'{data_name}-valid-ppi-blast-out')
    logger.info(F'GO: {len(mlb.classes_)}')
    logger.info(F'Training: {len(train_pid)}')
    logger.info(F'Validation: {len(valid_pid)}')

    # model and train
    model = Model(labels_num=len(mlb.classes_), dgl_graph=dgl_graph, network_x=pretrain_embedding,
                        input_size=len(pretrain_embedding[0]), pretrain_embedding=pretrain_embedding, pretrain_embedding2=pretrain_embedding,
                        interpro=interpro,interpro_size = interpro.shape[1],
                        model_path = Path(dataset_yaml['model_path'])/F'DeepFusionGO-Model-{model_id}-{data_name}', 
                        **model_yaml['model'])
    model.train((train_pid, train_go), (valid_pid, valid_go), **model_yaml['train'])


    # test
    mlb_test = get_go_mlb(Path(dataset_yaml['mlb']))
    test_cnf = dataset_yaml['test']
    test_pid_list, _ = get_pid_and_go(**test_cnf)
    test_res_idx_, _, test_ppi, _ = get_ppi_idx(True, test_pid_list, None ,
                                                        ppi_pid_map,test_cnf['fasta_file'], net_blastdb,
                                                        Path(dataset_yaml['results'])/F'{data_name}-test-ppi-blast-out')
    scores = np.zeros((len(test_pid_list), len(mlb_test.classes_)))
    scores[test_res_idx_] = model.predict(test_ppi, **model_yaml['test'])        
    res_path = Path(dataset_yaml['results'])/F'DeepFusionGO-Model-{model_id}-{data_name}-test'
    output_res(res_path.with_suffix('.txt'), test_pid_list, mlb_test.classes_, scores)
    np.save(res_path, scores)




def get_pid_and_go(fasta_file, pid_go_file=None, **kwargs):
    pid_list = []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split())[0]].append(line_list[1])
        return pid_list, [pid_go[pid_] for pid_ in pid_list]
    else:
        return pid_list ,None



def get_go_mlb(mlb_path: Path, labels=None) -> MultiLabelBinarizer:
    if mlb_path.exists():
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb



def get_ppi_idx(is_homo ,pid_list, data_y, ppi_pid_map, fasta_file=False, net_blastdb=False, blast_output_path=False):
    if is_homo:
        blast_sim = blast(net_blastdb, pid_list, fasta_file, blast_output_path)
        pid_list_ = []
        for i, pid in enumerate(pid_list):
            blast_sim[pid][None] = float('-inf')
            pid_ = pid if pid in ppi_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
            if pid_ is not None:
                pid_list_.append((i, pid, ppi_pid_map[pid_]))
        pid_list_ = tuple(zip(*pid_list_))
        pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    else:
        pid_list_ = tuple(zip(*[(i, pid, ppi_pid_map[pid])
                        for i, pid in enumerate(pid_list) if pid in ppi_pid_map]))
        pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))

    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def output_res(res_path: Path, pid_list, go_list, sc_mat):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                if s_ > 0.0:
                    print(pid_, go_, s_, sep='\t', file=fp)


if __name__ == '__main__':
    main()