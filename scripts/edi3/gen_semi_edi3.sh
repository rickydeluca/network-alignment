PD=dataspace/edi3
EDGEREMOVAL=(01 05 1 2)
# TRAINRATIO=(0.2 0.8)
TRAINRATIO=(0.8)

for ER in "${EDGEREMOVAL[@]}"; do
    for TR in "${TRAINRATIO[@]}"; do
        PREFIX=REGAL-d${ER}-seed1

        # Step 1: Generate target graph.
        python -m generate_dataset.semi_synthetic \
        --input_path ${PD} \
        --d 0.${ER} \
        --weighted

        # Step 2: Shuffle ID and index of nodes in target graph.
        DIR="${PD}/${PREFIX}" 
        python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 --weighted
        rm -r ${DIR} 
        mv ${DIR}--1 ${DIR}

        # Step 3: Split full dictionary into train and test files.
        python utils/split_dict.py \
        --input ${DIR}/dictionaries/groundtruth \
        --out_dir ${DIR}/dictionaries/ \
        --split ${TR}

        # Step4: Create features.
        PS=${PD}
        python -m utils.create_features \
        --input_data1 ${PS}/graphsage \
        --input_data2 ${PS}/${PREFIX}/graphsage \
        --feature_dim 300 \
        --ground_truth ${PS}/${PREFIX}/dictionaries/groundtruth

    done
done
