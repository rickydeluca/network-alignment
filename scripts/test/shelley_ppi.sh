PD=dataspace/ppi
PREFIX=REGAL-d2-seed1
TRAINRATIO=0.2

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
SHELLEY \
--cuda \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--batchsize 300 \
--num_epochs 10 \
--skip_train