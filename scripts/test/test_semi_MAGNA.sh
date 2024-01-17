#!/bin/bash

# Read command line arguments
DATA=$1
RUNS=$2
RAND=$3     # The random seed. If this value is greater than 0 it will be 
            # used as seed, otherwise a new random seed will be generated 
            # within each iteration.

# Set up test parameters
ALG_NAME="MAGNA"
DATA_SPACE="dataspace/$DATA"
TR=0.8          # Useless. Defined only because is required for `gen_semi.sh`.
EDGE_REMOVAL=("0" "01" "02" "03" "04" "05" "1" "2" "3" "5" "7" "9")
MEASURE="S3"
POPULATION_SIZE=(100 500 1000 2000 3000 5000 10000 15000)
NUM_GENERATIONS=(100 500 1000 1500 2000)
NUM_THREADS=8

# Init result dictionary
declare -A RESULTS_DICT

# RUN MAGNA
for PS in "${POPULATION_SIZE[@]}"; do
    for NG in "${NUM_GENERATIONS[@]}"; do

        # COMPUTE MEAN ACCURACY
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
                GROUNDTRUTH_PATH="--groundtruth ${DATA_SPACE}/${PREFIX}/dictionaries/groundtruth"

                # Configure algorithm
                ALG_ARGS="--source_edgelist ${DATA_SPACE}/edgelist/edgelist --target_edgelist ${DATA_SPACE}/${PREFIX}/edgelist/edgelist --measure $MEASURE --population_size $PS --num_generations $NG --num_threads $NUM_THREADS --outfile algorithms/MAGNA/output/magna --reverse"

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
            RESULTS_DICT["$ALG_NAME,0.$ER"]=$MEAN_ACCURACY,$STD_DEV_ACCURACY,$MEAN_EXECUTION_TIME

        done

        # Write the results to a CSV file
        if [ $RAND -le -1 ]; then
            RESULTS_CSV="results/${ALG_NAME}/${ALG_NAME}_${DATA}_p${PS}_g${NG}_n${RUNS}_seed_random.csv"
        else
            RESULTS_CSV="results/${ALG_NAME}/${ALG_NAME}_${DATA}_p${PS}_g${NG}_n${RUNS}_seed${RAND}.csv"
        fi
        echo "algorithm,edge_removal_ratio,mean_accuracy,std_dev,mean_execution_time" > $RESULTS_CSV
        for KEY in "${!RESULTS_DICT[@]}"; do
            echo "$KEY,${RESULTS_DICT[$KEY]}" >> $RESULTS_CSV
        done

    done
done