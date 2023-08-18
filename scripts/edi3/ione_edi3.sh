PD=dataspace/edi3
PREFIX=REGAL-d05-seed1
TRAINRATIO=0.8

# IONE
python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
IONE \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
# --cuda


