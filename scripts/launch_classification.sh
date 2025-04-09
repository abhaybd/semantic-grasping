if [ $# -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

NAME=$1
WORLD_SIZE=4

DATASET_NAME=0407_1606

gantry run -w ai2/abhayd -b ai2/prior \
    --name $NAME \
    --task-name $NAME \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --gpus $WORLD_SIZE \
    --weka prior-default:/data \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --shared-memory 128GiB \
    --allow-dirty \
    --install "pip install -r requirements-setup.txt ; pip install -r requirements.txt" \
    -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=$WORLD_SIZE src/train_classification.py \
        train.dataset.data_dir=/data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/observations \
        train.dataset.csv_path=/data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/dataset.csv \
        train.dataset.text_embedding_path=/data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/text_embeddings.npy
