from tqdm import tqdm, trange
import scipy.sparse as ssp
import numpy as np
import networkx as nx
import dgl
import click

@click.command()
@click.argument('ppi_network_path', type=click.Path(exists=True))
@click.argument('pre_graph_path', type=click.Path())

def main(ppi_network_path, pre_graph_path):
    ppi_degree =  ssp.load_npz(ppi_network_path)
    ppi_network = ppi_degree + ssp.eye(ppi_degree.shape[0], format='csr')
    
    # get top 100 
    rank = 100
    row, col, val = [], [], []
    for r in trange(ppi_network.shape[0]):
        for v, c in sorted(zip(ppi_network[r].data, ppi_network[r].indices), reverse=True)[:rank]:
            row.append(r)
            col.append(c)
            val.append(v)
    
    net_mat = ssp.csc_matrix((val, (row, col)), shape=ppi_network.shape).T
    degree_r = ssp.diags(np.asarray(net_mat.sum(0)).squeeze() ** -0.5, format='csr')
    degree_c = ssp.diags(np.asarray(net_mat.sum(1)).squeeze() ** -0.5, format='csr')
    ppi_network = degree_r @ net_mat @ degree_c

    # create graph by dgl
    ppi_net_mat_coo = ssp.coo_matrix(ppi_network)
    nx_ppi_graph = nx.DiGraph()
    for e1, e2, v in tqdm(zip(ppi_net_mat_coo.row, ppi_net_mat_coo.col, ppi_net_mat_coo.data), total=ppi_net_mat_coo.nnz):
        nx_ppi_graph.add_edge(e1, e2, ppi=v)
    dgl_graph = dgl.DGLGraph() 
    dgl_graph.from_networkx(nx_ppi_graph, edge_attrs=['ppi'])
    dgl.data.utils.save_graphs(pre_graph_path, dgl_graph)


if __name__ == '__main__':
    main()
