#!/bin/bash

# Read command line arguments
DATA=$1
ALG=$2
TR=$3
RUNS=$4
RAND=$5     # The random seed. If this value is greater than 0 it will be 
            # used as seed, otherwise a new random seed will be generated 
            # within each iteration.

# Set up test parameters
DATA_SPACE="dataspace/$DATA"
EDGE_REMOVAL=("0" "01" "02" "03" "04" "05" "1" "2" "3" "5" "7" "9")
# EDGE_REMOVAL=("0" "01")

if [ "$ALG" == "all" ]; then
    ALGORITHMS=("FINAL" "IsoRank" "BigAlign" "PALE" "IONE" "DeepLink" "HDA" "MAGNA")
else
    ALGORITHMS=("$ALG")
fi

# Init result dictionary
declare -A RESULTS_DICT

# COMPUTE MEAN ACCURACY
for ALG_NAME in "${ALGORITHMS[@]}"; do
    for ER in "${EDGE_REMOVAL[@]}"; do
        
        TOTAL_EXECUTION_TIME=0
        ACCURACY_ARRAY=()

        for ((i=1; i<=$RUNS; i++)); do
            # Generate a new semi-synthetic data
            if [ $RAND -le -1 ]; then
                SEED=$(python -c "import random; print(random.randint(0, 999))")
            else
                SEED=$RAND
            fi
            
            PREFIX="REGAL-d${ER}-seed$SEED"

            if [ "$DATA" == "edi3" ] || [ "$DATA" == "ppi" ]; then
                ./scripts/dataset/gen_semi.sh $DATA $TR $ER $SEED
            fi

            # Get groundtruth path
            if [[ "$ALGORITHM_NAME" =~ ^(FINAL|IsoRank|BigAlign|HDA|MAGNA)$ ]]; then
                GROUNDTRUTH_PATH="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/groundtruth"
            else
                GROUNDTRUTH_PATH="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.test.dict"
            fi

            # Configure algorithm
            if [ "$ALG_NAME" == "FINAL" ]; then
                ALG_ARGS=""
            elif [ "$ALG_NAME" == "IsoRank" ]; then
                ALG_ARGS=""
            elif [ "$ALG_NAME" == "BigAlign" ]; then
                ALG_ARGS=""
            elif [ "$ALG_NAME" == "PALE" ]; then
                ALG_ARGS="--train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            elif [ "$ALG_NAME" == "IONE" ]; then
                ALG_ARGS="--train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict"
            elif [ "$ALG_NAME" == "DeepLink" ]; then
                ALG_ARGS="--train_dict ${DATA_SPACE}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            elif [ "$ALG_NAME" == "HDA" ]; then
                VECTOR_SIZE=10000
                ALG_ARGS="--vector_size ${VECTOR_SIZE}"
            elif [ "$ALG_NAME" == "MAGNA" ]; then
                MEASURE="S3"
                POPULATION_SIZE=3000
                NUM_GENERATIONS=1000
                NUM_THREADS=8
                ALG_ARGS="--source_edgelist ${DATA_SPACE}/edgelist/edgelist --target_edgelist ${DATA_SPACE}/${PREFIX}/edgelist/edgelist --measure $MEASURE --population_size $POPULATION_SIZE --num_generations $NUM_GENERATIONS --num_threads $NUM_THREADS --outfile algorithms/MAGNA/output/magna --reverse"
            else
                echo "$ALG_NAME is not a valid algorithm."
            fi

            # Run alignment
            RESULT=$(python network_alignment.py \
                --source_dataset ${DATA_SPACE}/graphsage/ \
                --target_dataset ${DATA_SPACE}/${PREFIX}/graphsage/ \
                $GROUNDTRUTH_PATH \
                $ALG_NAME \
                $ALG_ARGS 2>&1)  # Capture both stdout and stderr

            echo "$RESULT"

            # Get accuracy and execution time
            EXECUTION_TIME=$(echo "$RESULT" | grep -oP 'Full_time: \K\d+\.\d+')
            ACCURACY=$(echo "$RESULT" | grep -oP 'Accuracy: \K\d+\.\d+')

            ACCURACY_ARRAY+=($ACCURACY)
            TOTAL_EXECUTION_TIME=$(echo "$TOTAL_EXECUTION_TIME + $EXECUTION_TIME" | bc)

            # In case of random seed remove the target network folder 
            # (a new one will be generated within the next iteration)
            if [ $RAND -le -1 ]; then
                rm -rf "${DATA_SPACE}/${PREFIX}"
            fi
        done

        # Compute means
        ACCURACY_LIST=$(IFS=,; echo "${ACCURACY_ARRAY[*]}")
        MEAN_ACCURACY=$(python -c "import statistics; accuracy_list = [$ACCURACY_LIST]; print(statistics.mean(accuracy_list))")
        STD_DEV_ACCURACY=$(python -c "import statistics; accuracy_list = [$ACCURACY_LIST]; print(statistics.stdev(accuracy_list))")
        MEAN_EXECUTION_TIME=$(echo "scale=4; $TOTAL_EXECUTION_TIME / $RUNS" | bc)
        RESULTS_DICT["$ALG_NAME,0.$ER,$TR"]=$MEAN_ACCURACY,$STD_DEV_ACCURACY,$MEAN_EXECUTION_TIME

    done
done


# Write the results to a CSV file
if [ $RAND -le -1 ]; then
    RESULTS_CSV="results/${ALG}_${DATA}_s${TR}_n${RUNS}_seed_random.csv"
else
    RESULTS_CSV="results/${ALG}_${DATA}_s${TR}_n${RUNS}_seed${RAND}.csv"
fi
echo "algorithm,edge_removal_ratio,train_ratio,mean_accuracy,std_dev,mean_execution_time" > $RESULTS_CSV
for KEY in "${!RESULTS_DICT[@]}"; do
    echo "$KEY,${RESULTS_DICT[$KEY]}" >> $RESULTS_CSV
done