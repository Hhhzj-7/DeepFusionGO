# DeepFusionGO
DeepFusionGO: Protein function prediction by fusing heterogeneous features through deep learning

We use the dataset and some code from DeepGraphGO(https://github.com/yourh/DeepGraphGO).

## Environment

environment.yml

## Train and Valid
mf: 
python -u main.py -d settings/mf.yaml --model-id 0
python -u main.py -d settings/mf.yaml --model-id 1
python -u main.py -d settings/mf.yaml --model-id 2


bp:
python -u main.py -d settings/bp.yaml --model-id 0
python -u main.py -d settings/bp.yaml --model-id 1
python -u main.py -d settings/bp.yaml --model-id 2


cc:
python -u main.py -d settings/cc.yaml --model-id 0
python -u main.py -d settings/cc.yaml --model-id 1
python -u main.py -d settings/cc.yaml --model-id 2

## Test
mf:
python bagging.py -m settings/model.yaml -d settings/mf.yaml -n 3
python evaluation.py data/mf_test_go.txt results/DeepFusionGO-Ensemble-mf-test.txt


bp:
python bagging.py -m settings/model.yaml -d settings/bp.yaml -n 3
python evaluation.py data/bp_test_go.txt results/DeepFusionGO-Ensemble-bp-test.txt


cc:
python bagging.py -m settings/model.yaml -d settings/cc.yaml -n 3
python evaluation.py data/cc_test_go.txt results/DeepFusionGO-Ensemble-cc-test.txt