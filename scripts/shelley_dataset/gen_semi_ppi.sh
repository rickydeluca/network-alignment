# Script to generate a dataset composed by semi-syntethic
# target networks for graph-matching starting from a 
# real source graph.
#
# Each target graph is obtained from the source graph by
# randomly permuting its nodes and by adding some noise.
#
# The noise is added by subtraction or by addition:
#   - subtraction:      drop some edges with probability
#   - addition:         add some "dummy" edges with probability

INPUT_DIR=dataspace/ppi                 # Source network
P_ADDS=(0.00 0.05 0.10 0.15 0.20 0.25)  # Probabilities for edge addition
P_RMVS=(0.00 0.05 0.10 0.15 0.20 0.25)  # Probabilities for edge dropping
N_COPIES=100                            # How many copies for each probability combination
SEED=42

for P_ADD in "${P_ADDS[@]}"; do
    for P_RM in "${P_RMVS[@]}"; do
        for ((IDX=1; IDX<=N_COPIES; IDX++)); do
            # 0. Update seed and define output directory
            ((SEED++))
            TARGET_DIR="${INPUT_DIR}/targets/add${P_ADD#0.}_rm${P_RM#0.}_${IDX}"

            # 1. Target graph generation
            python -m generate_dataset.semi_synthetic_shelley \
            --input_path $INPUT_DIR \
            --idx $IDX \
            --p_add $P_ADD \
            --p_rm $P_RM \
            --seed $SEED \
            --outdir $TARGET_DIR

            # 2. Shuffle ID and index of nodes in target graph
            python -m utils.shuffle_graph_shelley \
            --input_dir ${TARGET_DIR} \
            --outdir ${TARGET_DIR}--${SEED} \
            --seed ${SEED} \
            --n_shuffles 100

            rm -r ${TARGET_DIR}
            mv ${TARGET_DIR}--${SEED} ${TARGET_DIR}
        done    
    done
done