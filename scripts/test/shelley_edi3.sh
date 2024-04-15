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
--train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--root_dir dataspace/edi3 \
--head common \
--backbone gin \
--loss_func distill_qc \
--batchsize 4 \
--p_add 0.1 \
--p_rm 0.0 \
--alpha 0.4 \
--optimizer adam \
--use_scheduler \
--train \
--validate \
--test \
--split_ratio 0.2 \
--size 100 \
--self_supervised