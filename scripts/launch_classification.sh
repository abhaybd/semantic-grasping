if [ $# -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

NAME=$1
DATASET_NAME=dataset_0311_1749

OBS_DATASET=$(python -m semantic_grasping_datagen.get_data_for_dataset abhayd/${DATASET_NAME})
if [ $? -ne 0 ]; then
    echo "Failed to get dataset"
    exit 1
fi

gantry run -w ai2/abhayd -b ai2/prior \
    --name $NAME \
    --task-name $NAME \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --gpus 4 \
    --dataset abhayd/$DATASET_NAME:/dataset \
    --dataset $OBS_DATASET:/data \
    --priority normal \
    --cluster ai2/augusta-google-1 \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/prior-elanding \
    --shared-memory 128GiB \
    --allow-dirty \
    -- \
    python src/train_classification.py
