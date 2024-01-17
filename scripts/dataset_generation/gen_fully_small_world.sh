# This script generates a series of fully-synthetic datasets 
# following the small world model with different number of nodes
# and average degrees.

NUM_NODES=(1700 1750 1800 1850 1900 1950 2000)
AVG_DEGREES=(20 25 30)
REMOVAL_RATIOS=(0 01 02 03 04 05 06 07 08 09 1 2 3 4 5)

for NN in "${NUM_NODES[@]}"; do 
    for aver in "${AVG_DEGREES[@]}"; do
        # Step 1: generate a graph.
        python -m generate_dataset.fully_synthetic \
        --output_path dataspace/fully-synthetic \
        --model small_world \
        --n ${NN} \
        --aver ${aver} \
        --feature_dim 300   # Generate input features only once and derive the target features from them.

        for ER in "${REMOVAL_RATIOS[@]}"; do
            # Step 2: run semi-synthetic to create target noise graph.
            python -m generate_dataset.semi_synthetic \
            --input_path dataspace/fully-synthetic/small_world-n${NN}-p${aver} \
            --d 0.${ER} 

            # Step 3: shuffle id and index of nodes in target graph.
            # Dictionaries will be saved at dataspace/fully_synthetic/small_world-n${NN}-p${aver}/REGAL-d${ER}-seed1/dictionaries/groundtruth
            DIR="dataspace/fully-synthetic/small_world-n${NN}-p${aver}/REGAL-d${ER}-seed1" 
            python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--1 
            rm -r ${DIR} 
            mv ${DIR}--1 ${DIR}


            # Step 4: split full dictionary into train and test files.
            python utils/split_dict.py \
            --input ${DIR}/dictionaries/groundtruth \
            --out_dir ${DIR}/dictionaries/ \
            --split 0.8


            # Step 5 [optioinal]: create features for dataset.
            PS="dataspace/fully-synthetic/small_world-n${NN}-p${aver}" 
            python -m utils.create_features \
            --input_data1 ${PS}/graphsage \
            --input_data2 ${PS}/REGAL-d${ER}-seed1/graphsage \
            --feature_dim 300 \
            --ground_truth ${PS}/REGAL-d${ER}-seed1/dictionaries/groundtruth \
            --only_target
        done
    done 
done 


# After 4 steps, the source_dataset path is: dataspace/fully-synthetic/small_world-n${NN}-p${aver}/graphsage 
# and target dataset path is: dataspace/fully-synthetic/small_world-n${NN}-p${aver}/REGAL-d${ER}-seed1/graphsage
# full and train, test dictionary can be found at dataspace/fully-synthetic/small_world-n${NN}-p${aver}/REGAL-d${ER}-seed1/dictionaries/