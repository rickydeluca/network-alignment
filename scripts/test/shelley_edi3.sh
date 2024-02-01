# Modify here:
DATA=edi3
TRAINRATIO=0.2
ERR=15          # Edge Removal Ratio
SEED=1

# Generate dataset
./scripts/dataset/gen_semi.sh $DATA $TRAINRATIO $ERR $SEED

# Run algorithm
PD=dataspace/${DATA}
PREFIX=REGAL-d${ERR}-seed${SEED}

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
SHELLEY \
--cuda \
--root_dir dataspace/edi3 \
--p_add 0.2 \
--p_rm 0.2 \
--lr 0.000003 \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--optimizer sgd \
--use_scheduler