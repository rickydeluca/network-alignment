PD=dataspace/edi3
PREFIX=REGAL-d2-seed1
TRAINRATIO=0.8

# SANE
python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
SANE \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--embedding_model sage \
--mapping_model inner_product \
--device cuda \
--epochs 100 \
--lr 0.00005 \
--num_layers 1 \
--early_stop \