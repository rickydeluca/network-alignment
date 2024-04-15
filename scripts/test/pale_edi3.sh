# modify here:
DATA=edi3
TRAINRATIO=0.2
ERR=5        # edge removal ratio
SEED=1

# generate dataset
./scripts/dataset/gen_semi.sh $DATA $TRAINRATIO $ERR $SEED

# run algorithm
PD=dataspace/${DATA}
PREFIX=REGAL-d${ERR}-seed${SEED}

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict \
PALE \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--cuda \