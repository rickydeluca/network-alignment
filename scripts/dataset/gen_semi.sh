# Script to generate a semi-synthetic dataset from a real one.
# To run it put yourself in the main folder and use:
#
# $ ./script/dataset/gen_semi.sh <dataset_name> <train_ratio> <seed_value>
#
# The command line arguments are:
#       dataset_name:       The name of the original network from which generate the
#                           semi-synthetic dataset. Choose from: 'edi3' and 'ppi'.
#
#       train_ratio:        The train-test split ratio used by the representation
#                           learning algorithms. For example '0.8' 
#                           means 80% training, 20% test.
#
#       edge_removal_ratio: The percentage of edges to remove in the target
#                           semi-synthetic network.
#
#       seed:               The seed value used for the randomic operations.



PD=dataspace/$1     # Inupt networks
TR=$2               # Train-test split ratio    
ER=$3               # Edge removal ratio
SEED=$4

PREFIX=REGAL-d${ER}-seed${SEED}

# Step 1: Generate target graph
if [ "$1" == "edi3" ]; then
    python -m generate_dataset.semi_synthetic \
    --input_path ${PD} \
    --d 0.${ER} \
    --seed $SEED \
    --weighted
else
    python -m generate_dataset.semi_synthetic \
    --input_path ${PD} \
    --d 0.${ER} \
    --seed $SEED
fi

# Step 2: Shuffle ID and index of nodes in target graph
DIR="${PD}/${PREFIX}"

if [ "$1" == "edi3" ]; then
    python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--${SEED} --weighted
else
    python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}--${SEED}
fi

rm -r ${DIR} 
mv ${DIR}--${SEED} ${DIR}

# Step 3: Split full dictionary into train and test files
python utils/split_dict.py \
--input ${DIR}/dictionaries/groundtruth \
--out_dir ${DIR}/dictionaries/ \
--split ${TR}

# Step 4: Create features
PS=${PD}
python -m utils.create_features \
--input_data1 ${PS}/graphsage \
--input_data2 ${PS}/${PREFIX}/graphsage \
--feature_dim 1024 \
--ground_truth ${PS}/${PREFIX}/dictionaries/groundtruth