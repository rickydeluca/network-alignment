#!/bin/bash

# Read command line arguments
DATA=$1
ALG=$2
TR=$3
RUNS=$4

# Set up test parameters
DATA_SPACE="dataspace/$DATA"
EDGE_REMOVAL=("0" "01" "02" "03" "04" "05" "1" "2" "3" "5" "7" "9")
SEED=42

# Init result dictionary
declare -A RESULTS_DICT

# COMPUTE MEAN ACCURACY
for ER in "${EDGE_REMOVAL[@]}"; do
    PREFIX="REGAL-d${ER}-seed${SEED}"  # Target network directory

    # Define algorithm configurations
    if [ "$ALG" == "final" ]; then
        ALGORITHMS=("FINAL")

    elif [ "$ALG" == "isorank" ]; then
        ALGORITHMS=("IsoRank")

    elif [ "$ALG" == "bigalign" ]; then
        ALGORITHMS=("BigAlign")

    elif [ "$ALG" == "pale" ]; then
        ALGORITHMS=("PALE --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda")

    elif [ "$ALG" == "ione" ]; then
        ALGORITHMS=("IONE --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict")

    elif [ "$ALG" == "deeplink" ]; then
        ALGORITHMS=("DeepLink --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda")

    elif [ "$ALG" == "hda" ]; then
        VECTOR_SIZE=10000
        ALGORITHMS=("HDA --vector_size ${VECTOR_SIZE}")

    elif [ "$ALG" == "magna" ]; then
        MEASURE="S3"
        POPULATION_SIZE=3000
        NUM_GENERATIONS=1000
        NUM_THREADS=8

        ALGORITHMS=("MAGNA --source_edgelist ${DATA_SPACE}/edgelist/edgelist --target_edgelist ${DATA_SPACE}/${PREFIX}/edgelist/edgelist --measure $MEASURE --population_size $POPULATION_SIZE --num_generations $NUM_GENERATIONS --num_threads $NUM_THREADS --outfile algorithms/MAGNA/output/magna --reverse")

    else
        VECTOR_SIZE=10000
        MEASURE="S3"
        POPULATION_SIZE=15000
        NUM_GENERATIONS=2000
        NUM_THREADS=8

        ALGORITHMS=(
            "FINAL"
            "IsoRank"
            "BigAlign"
            "PALE --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            "IONE --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict"
            "DeepLink --train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            "HDA --vector_size ${VECTOR_SIZE}"
            "MAGNA --source_edgelist ${DATA_SPACE}/edgelist/edgelist --target_edgelist ${DATA_SPACE}/${PREFIX}/edgelist/edgelist --measure $MEASURE --population_size $POPULATION_SIZE --num_generations $NUM_GENERATIONS --num_threads $NUM_THREADS --outfile algorithms/MAGNA/output/magna --reverse"
        )
    fi

    for ALGO_CONFIG in "${ALGORITHMS[@]}"; do

        # Split the algorithm configuration to get the algorithm name and its arguments
        ALGORITHM_NAME="$ALGO_CONFIG"
        ALGORITHM_ARGS=""
        if [[ "$ALGO_CONFIG" == *" "* ]]; then
            IFS=" " read -r ALGORITHM_NAME ALGORITHM_ARGS <<< "$ALGO_CONFIG"
        fi

        echo "Algorithm: ${ALGORITHM_NAME}"
        echo "Arguments: ${ALGORITHM_ARGS}"

        TOTAL_EXECUTION_TIME=0
        ACCURACY_ARRAY=()

        for ((i=1; i<=$RUNS; i++)); do

            # Generate a new semi-synthetic network for each run
            if [ "$DATA" == "edi3" ] || [ "$DATA" == "ppi" ]; then
                ./scripts/dataset/gen_semi.sh $DATA $TR $ER $SEED
            fi
            
            # Get path of the groundtruth alignment
            if [[ "$ALGORITHM_NAME" =~ ^(FINAL|IsoRank|BigAlign)$ ]]; then
                GROUNDTRUTH_ARG="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/groundtruth"
            else
                GROUNDTRUTH_ARG="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.test.dict"
            fi

            # Export alignment
            # if [[ "$ALGORITHM_NAME" =~ ^(FINAL|IsoRank|BigAlign)$ ]]; then
            #     EXPORT_ARG="--alignment_matrix_name ${DATA}-${ALGORITHM_NAME}-d${ER}.csv --transpose_alignment_matrix"
            # else
            #     EXPORT_ARG="--alignment_matrix_name ${DATA}-${ALGORITHM_NAME}-s${TR}-d${ER}.csv --transpose_alignment_matrix"
            # fi
                
            # Run network alignment script and capture accuracy and execution time
            RESULT=$(python network_alignment.py \
                --source_dataset ${DATA_SPACE}/graphsage/ \
                --target_dataset ${DATA_SPACE}/${PREFIX}/graphsage/ \
                $GROUNDTRUTH_ARG \
                $ALGORITHM_NAME \
                $ALGORITHM_ARGS 2>&1)  # Capture both stdout and stderr

            echo "$RESULT"

            # Capture and accumulate output values
            EXECUTION_TIME=$(echo "$RESULT" | grep -oP 'Full_time: \K\d+\.\d+')
            ACCURACY=$(echo "$RESULT" | grep -oP 'Accuracy: \K\d+\.\d+')

            ACCURACY_ARRAY+=($ACCURACY)
            TOTAL_EXECUTION_TIME=$(echo "$TOTAL_EXECUTION_TIME + $EXECUTION_TIME" | bc)
            
        done


        # Compute and store mean accuracy and mean execution time
        ACCURACY_LIST=$(IFS=,; echo "${ACCURACY_ARRAY[*]}")
        MEAN_ACCURACY=$(python -c "import statistics; accuracy_list = [$ACCURACY_LIST]; print(statistics.mean(accuracy_list))")
        STD_DEV_ACCURACY=$(python -c "import statistics; accuracy_list = [$ACCURACY_LIST]; print(statistics.stdev(accuracy_list))")
        MEAN_EXECUTION_TIME=$(echo "scale=4; $TOTAL_EXECUTION_TIME / $RUNS" | bc)
        RESULTS_DICT["$ALGORITHM_NAME,0.$ER,$TR"]=$MEAN_ACCURACY,$STD_DEV_ACCURACY,$MEAN_EXECUTION_TIME

    done
done

# Write the results to a CSV file
RESULTS_CSV="results/${ALG}_${DATA}_s${TR}_n${RUNS}_seed${SEED}.csv"
echo "algorithm,edge_removal_ratio,train_ratio,mean_accuracy,std_dev,mean_execution_time" > $RESULTS_CSV
for KEY in "${!RESULTS_DICT[@]}"; do
    echo "$KEY,${RESULTS_DICT[$KEY]}" >> $RESULTS_CSV
done