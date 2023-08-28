PD=dataspace/edi3
TRAINRATIO=(0.8)
EDGEREMOVAL=(01 05 1 2)

# Initialize a dictionary to hold the results
declare -A results_dict

for ER in "${EDGEREMOVAL[@]}"; do
    for TR in "${TRAINRATIO[@]}"; do
        PREFIX=REGAL-d${ER}-seed1

        algorithms=(
            "FINAL "        # The space is important
            "IsoRank "      # ""
            "BigAlign "     # ""
            "PALE --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
            "IONE --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict"
            "DeepLink --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict --cuda"
        )

        for algo_config in "${algorithms[@]}"; do
            algorithm_name=$(echo "$algo_config" | cut -d ' ' -f 1)
            algorithm_args=$(echo "$algo_config" | cut -d ' ' -f 2-)

            echo "name: ${algorithm_name}"
            echo "args: ${$algorithm_args}"
            
            if [[ "$algorithm_name" == "FINAL" || "$algorithm_name" == "IsoRank" || "$algorithm_name" == "BigAlign" ]]; then
                groundtruth_arg="--groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth"
            else
                groundtruth_arg="--groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict"
            fi

            python network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX}/graphsage/ \
                $groundtruth_arg \
                $algorithm_name \
                $algorithm_args >> temp/output.txt

            accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
            results_dict["$algorithm_name,$ER,$TR"]=$accuracy

            rm temp/output.txt
        done
    done
done

# Write the results to a CSV file
echo "Algorithm,Edge_Removal_Ratio,Train_Ratio,Accuracy" > results/edi3_all_results.csv
for key in "${!results_dict[@]}"; do
    echo "$key,${results_dict[$key]}" >> results/edi3_all_results.csv
done
