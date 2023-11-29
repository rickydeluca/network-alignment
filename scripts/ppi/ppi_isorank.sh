PD=dataspace/ppi
PREFIX=REGAL-d05-seed1

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX}/graphsage/ \
--groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
--alignment_matrix_name "ppi_isorank_d05.csv" \
--transpose_alignment_matrix \
IsoRank 

