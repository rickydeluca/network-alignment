PD=dataspace/edi3
PREFIX=REGAL-d05-seed1
TRAINRATIO=0.8

# SANE
python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
SANE \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--embedding_model sage \
--mapping_model cosine_similarity \
--device cuda \
--epochs 100 \
--early_stop \