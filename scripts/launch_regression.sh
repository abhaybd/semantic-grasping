if [ $# -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

NAME=$1
DATASET_NAME=dataset_0401_2018
WORLD_SIZE=4

OBS_DATASET=$(python -m semantic_grasping_datagen.get_data_for_dataset abhayd/${DATASET_NAME})
if [ $? -ne 0 ]; then
    echo "Failed to get dataset"
    exit 1
fi

gantry run -w ai2/abhayd -b ai2/prior \
    --name $NAME \
    --task-name $NAME \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --gpus $WORLD_SIZE \
    --dataset abhayd/$DATASET_NAME:/dataset \
    --dataset $OBS_DATASET:/data \
    --priority normal \
    --cluster ai2/augusta-google-1 \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --shared-memory 128GiB \
    --allow-dirty \
    -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=$WORLD_SIZE src/train_regression.py
