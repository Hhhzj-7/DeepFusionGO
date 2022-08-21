
"""

From "https://github.com/yourh/DeepGraphGO"

"""

import click
from itertools import chain
from collections import defaultdict
import scipy.sparse as ssp

from deepfusiongo.evaluation import fmax, pair_aupr, ROOT_GO_TERMS


def evaluate_metrics(pid_go, pid_go_sc):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc) as fp:
        for line in fp:
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list:=line.split('\t'))[2])
    pid_go_sc = dict(pid_go_sc)

    pid_go = defaultdict(list)
    with open(pid_go) as fp:
        for line in fp:
            pid_go[(line_list:=line.split('\t'))[0]].append(line_list[1])
    pid_go = dict(pid_go)

    pid_list = list(pid_go.keys())
    go_list = sorted(set(list(chain(*([pid_go[p_] for p_ in pid_list] +
                                      [pid_go_sc[p_] for p_ in pid_list if p_ in pid_go_sc])))) - ROOT_GO_TERMS)
    go_mat, score_mat = get_pid_go_mat(pid_go, pid_list, go_list), get_pid_go_sc_mat(pid_go_sc, pid_list, go_list)
    return fmax(go_mat, score_mat), pair_aupr(go_mat, score_mat)


def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))


def get_pid_go_sc_mat(pid_go_sc, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_sc:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_sc[pid_].get(go_, -1e100)
    return sc_mat



@click.command()
@click.argument('pid_go', type=click.Path(exists=True))
@click.argument('pid_go_sc', type=click.Path(exists=True))
def main(pid_go, pid_go_sc):
    (fmax_, t_), aupr_ = evaluate_metrics(pid_go, pid_go_sc)
    print(F'Fmax: {fmax_:.3f} {t_:.2f}', F'AUPR: {aupr_:.3f}')


if __name__ == '__main__':
    main()
