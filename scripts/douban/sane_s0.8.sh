PD=dataspace/douban
PREFIX1=online
PREFIX2=offline
TRAINRATIO=0.8

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
SANE \
--train_dict ${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--embedding_model spectral \
--device cuda \