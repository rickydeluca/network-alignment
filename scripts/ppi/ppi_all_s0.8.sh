#!/bin/bash

# Declare input parameters
DATA="ppi"
DATA_SPACE="dataspace/$DATA"
TRAIN_RATIO=("0.8")
EDGE_REMOVAL=("0" "01" "05" "1" "2")

# Initialize results dictionary
declare -A RESULTS_DICT

# Loop through different edge removal ratios
for ER in "${EDGE_REMOVAL[@]}"; do
    # Loop through different train ratios
    for TR in "${TRAIN_RATIO[@]}"; do
        PREFIX="REGAL-d${ER}-seed1"

        # Define algorithm configurations
        ALGORITHMS=(
            "FINAL"
            "IsoRank"
            "BigAlign"
            "PALE --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            "IONE --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict"
            "DeepLink --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            "HDA --vector_size 10000"
        )

        # Loop through algorithm configurations
        for ALGO_CONFIG in "${ALGORITHMS[@]}"; do
            # Split algorithm configuration into name and arguments
            ALGORITHM_NAME="$ALGO_CONFIG"
            ALGORITHM_ARGS=""
            if [[ "$ALGO_CONFIG" == *" "* ]]; then
                IFS=" " read -r ALGORITHM_NAME ALGORITHM_ARGS <<< "$ALGO_CONFIG"
            fi

            # Print algorithm details
            echo "Algorithm: ${ALGORITHM_NAME}"
            echo "Arguments: ${ALGORITHM_ARGS}"
            
            # Set ground truth argument based on algorithm
            if [[ "$ALGORITHM_NAME" =~ ^(FINAL|IsoRank|BigAlign)$ ]]; then
                GROUNDTRUTH_ARG="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/groundtruth"
            else
                GROUNDTRUTH_ARG="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.test.dict"
            fi

            # Export alignment args
            if [[ "$ALGORITHM_NAME" =~ ^(FINAL|IsoRank|BigAlign)$ ]]; then
                EXPORT_ARG="--alignment_matrix_name ${DATA}_${ALGORITHM_NAME}_d${ER}.csv --transpose_alignment_matrix"
            else
                EXPORT_ARG="--alignment_matrix_name ${DATA}_${ALGORITHM_NAME}_s${TR}_d${ER}.csv --transpose_alignment_matrix"
            fi
            

            # Run network alignment script and capture accuracy
            ACCURACY=$(python network_alignment.py \
                --source_dataset ${DATA_SPACE}/graphsage/ \
                --target_dataset ${DATA_SPACE}/${PREFIX}/graphsage/ \
                $GROUNDTRUTH_ARG \
                $EXPORT_ARG \
                $ALGORITHM_NAME \
                $ALGORITHM_ARGS | grep -oP 'Accuracy: \K\d+\.\d+')

            # Store accuracy in results dictionary
            RESULTS_DICT["$ALGORITHM_NAME,0.$ER,$TR"]=$ACCURACY
        done
    done
done

# Write results to CSV
RESULTS_CSV="results/ppi_all_split0.8.csv"
echo "Algorithm,Edge_Removal_Ratio,Train_Ratio,Accuracy" > $RESULTS_CSV
for KEY in "${!RESULTS_DICT[@]}"; do
    echo "$KEY,${RESULTS_DICT[$KEY]}" >> $RESULTS_CSV
done
