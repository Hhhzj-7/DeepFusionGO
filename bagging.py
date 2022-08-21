
"""

From "https://github.com/yourh/DeepGraphGO"

"""

import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('-n', 'num_models', type=click.INT, default=None)
def main(data_cnf, model_cnf, num_models):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    data_name, model_name = data_cnf['dataset'], model_cnf['name']
    res_path = Path(data_cnf['results'])
    mlb = get_mlb(Path(data_cnf['mlb']))
    test_cnf = data_cnf['test']
    test_name = test_cnf.pop('name')
    with open(test_cnf['pid_list_file']) as fp:
        test_pid_list = [line.split()[0] for line in fp]
    sc_mat = np.zeros((len(test_pid_list), len(mlb.classes_)))
    for i in range(num_models):
        sc_mat += np.load(res_path/F'{model_name}-Model-{i}-{data_name}-{test_name}.npy') / num_models
    res_path_ = res_path/F'{model_name}-Ensemble-{data_name}-{test_name}'
    np.save(res_path_, sc_mat)
    output_res(res_path_.with_suffix('.txt'), test_pid_list, mlb.classes_, sc_mat)


def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    if mlb_path.exists():
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb

def output_res(res_path: Path, pid_list, go_list, sc_mat):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                if s_ > 0.0:
                    print(pid_, go_, s_, sep='\t', file=fp)

if __name__ == '__main__':
    main()
