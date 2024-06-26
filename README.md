# DeepFusionGO
DeepFusionGO: Protein function prediction by fusing heterogeneous features through deep learning


We extend the dataset from DeepGraphGO(https://github.com/yourh/DeepGraphGO). 


All the data we use can be download from [data](https://drive.google.com/file/d/1Wtb-i0NP2IjvMhsa2KVA_OXn0GZ0scn5/view?usp=drive_link).

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
