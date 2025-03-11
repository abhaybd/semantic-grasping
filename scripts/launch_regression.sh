if [ $# -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

NAME=$1
DATASET_NAME=dataset_0310_1749
OBS_DATASET_NAME=obsgen_0310_1650

gantry run -w ai2/abhayd -b ai2/prior \
    --name $NAME \
    --task-name $NAME \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --gpus 2 \
    --dataset $DATASET_NAME=/dataset \
    --dataset $OBS_DATASET_NAME=/data \
    -- \
    python src/train_regression.py
