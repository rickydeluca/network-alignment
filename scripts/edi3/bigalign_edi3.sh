PD=dataspace/edi3
PREFIX=REGAL-d05-seed1
TRAINRATIO=0.2

# BigAlign
python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
BigAlign
