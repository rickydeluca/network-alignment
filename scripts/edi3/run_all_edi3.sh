PD=dataspace/edi3
# TRAINRATIO=(0.2 0.8)
TRAINRATIO=(0.8)
EDGEREMOVAL=(01 05 1 2)

for ER in "${EDGEREMOVAL[@]}"; do
    for TR in "${TRAINRATIO[@]}"; do
                
        PREFIX=REGAL-d${ER}-seed1

        # REGAL
        # algorithm_name="FINAL"  
        # python network_alignment.py \
        # --source_dataset ${PD}/graphsage/ \
        # --target_dataset ${PD}/${PREFIX}/graphsage/ \
        # --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        # REGAL >> temp/output.txt
        # accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        # echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        # rm temp/output.txt

        # FINAL
        algorithm_name="FINAL"  
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        FINAL >> temp/output.txt
        accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        rm temp/output.txt

        # IsoRank
        algorithm_name="IsoRank" 
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        IsoRank >> temp/output.txt
        accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        rm temp/output.txt

        # BigAlign
        algorithm_name="BigAlign" 
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/groundtruth \
        BigAlign >> temp/output.txt
        accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        rm temp/output.txt

        # PALE
        algorithm_name="PALE" 
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict \
        PALE \
        --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict \
        --cuda >> temp/output.txt
        accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        rm temp/output.txt

        # IONE
        algorithm_name="IONE" 
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict \
        IONE \
        --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict >> temp/output.txt
        accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        rm temp/output.txt

        # DeepLink
        algorithm_name="DeepLink" 
        python network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX}/graphsage/ \
        --groundtruth ${PD}/${PREFIX}/dictionaries/node,split=${TR}.test.dict \
        DeepLink \
        --train_dict ${PD}/${PREFIX}/dictionaries/node,split=${TR}.train.dict \
        --cuda >> temp/output.txt
        accuracy=$(grep -oP 'Accuracy: \K\d+\.\d+' temp/output.txt)
        echo "$algorithm_name,$accuracy" >> results/edi3_er${ER}_tr${TR}_accuracy.csv
        rm temp/output.txt

    done
done


