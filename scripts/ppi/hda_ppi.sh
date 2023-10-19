PD=dataspace/ppi
PREFIX=REGAL-d05-seed1
TRAINRATIO=0.8

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
HDA \
--vector_size 10000

