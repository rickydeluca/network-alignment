PD=dataspace/ppi
PREFIX=REGAL-d05-seed1
TRAINRATIO=0.8

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
SANE \
--embedding "sage" \
--prediction "dnn" \
--lr 0.0003 \
--epochs 100 \
--hidden_size 64 \
--output_size 64 \
--num_layers 2 \
--batch_size 256 