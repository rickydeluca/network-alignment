PD=dataspace/ppi
PREFIX=REGAL-d2-seed1
TRAINRATIO=0.2

# SANE
python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
SANE \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--embedding_model sage \
--device cuda \
--batch_size_emb 4096 \
--early_stop_emb