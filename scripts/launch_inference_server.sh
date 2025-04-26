#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model>"
    echo "  model: 'graspmolmo' or 'molmo'"
    exit 1
fi

MODEL=$1

if [ "$MODEL" = "graspmolmo" ]; then
    CKPT_DIR="/weka/oe-training-default/roseh/mm_olmo/robomolmo_checkpoints/graspmolmo_cotraining_06_graspmolmo-focused_20250423_014047/latest-unsharded"
elif [ "$MODEL" = "molmo" ]; then
    CKPT_DIR="/weka/oe-training-default/roseh/molmo_pretrained_checkpoints/Molmo-7B-D-0924-Pretrained"
else
    echo "Error: MODEL must be either 'graspmolmo' or 'molmo'"
    exit 1
fi

beaker session create -w ai2/abhayd --budget ai2/prior \
    --name "molmo_tmp_inference_server" \
    --mount src=weka,ref=prior-default,dst=/weka/prior \
    --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
    --env CKPT_DIR=$CKPT_DIR \
    --workdir /weka/prior/abhayd/semantic-grasping \
    --detach \
    --bare \
    --image beaker://ai2/cuda11.8-dev-ubuntu20.04 \
    --priority high \
    --gpus 1 \
    --cluster ai2/neptune-cirrascale \
    --port 8080 \
    -- \
    /weka/prior/abhayd/envs/grasping/bin/fastapi run semantic_grasping/eval/inference_server.py --port 8080
