PD=dataspace/ppi
PREFIX=REGAL-d05-seed1
TRAINRATIO=0.2

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
COMMON \
--cuda \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--emb_epochs 1