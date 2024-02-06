# modify here:
DATA=ppi
TRAINRATIO=0.8
ERR=05        # edge removal ratio
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
SHELLEY \
--cuda \
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--root_dir dataspace/ppi \
--head stablegm \
--backbone gin \
--loss_func cml \
--batchsize 2 \
--p_add 0.0 \
--p_rm 0.0 \
--alpha 0.4 \
--optimizer adam \
--use_scheduler \
--train \
--validate \
--test
