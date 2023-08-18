PD=dataspace/edi3
TRAINRATIO=(0.2 0.8)
EDGEREMOVAL=(01 05 1 2)

for ER in "${EDGEREMOVAL[@]}"; do
    for TR in "${TRAINRATIO[@]}"; do
                
        PREFIX=REGAL-d${ER}-seed1

        # REGAL
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        REGAL 

        # FINAL 
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        FINAL

        # IsoRank
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        IsoRank

        # BigAlign
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        BigAlign

        # PALE
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict \
        PALE \
        --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict \
        --cuda

        # IONE
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict \
        IONE \
        --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict \
        --cuda

        # DeepLink
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict \
        DeepLink \
        --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict \
        --cuda 

    done
done


