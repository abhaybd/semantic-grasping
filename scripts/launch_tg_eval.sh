TG_DIR=/weka/prior/abhayd/semantic-grasping-datasets/taskgrasp_image
MODEL=molmo

if [ "$MODEL" = "graspmolmo" ]; then
    MODEL_ARGS="--grasp-molmo /weka/oe-training-default/roseh/mm_olmo/robomolmo_checkpoints/graspmolmo_cotraining_06_graspmolmo-focused_20250423_014047/latest-unsharded"
elif [ "$MODEL" = "molmo" ]; then
    MODEL_ARGS="--molmo"
else
    echo "Unknown model: $MODEL"
    exit 1
fi


TIMESTAMP=$(date +%m%d_%H%M)
OUT_DIR=/weka/prior/abhayd/semantic-grasping-datasets/eval_${TIMESTAMP}

SPLITS="o t"

for split in $SPLITS; do
    gantry run -w ai2/abhayd -b ai2/prior \
        --name "${MODEL}_tg_eval_${split}_${TIMESTAMP}" \
        --task-name "${MODEL}_tg_eval_${split}_${TIMESTAMP}" \
        --env-secret GITHUB_TOKEN=GITHUB_TOKEN \
        --dataset-secret SSH_KEY:/root/.ssh/id_ed25519 \
        --beaker-image ai2/cuda11.8-dev-ubuntu20.04 \
        --gpus 1 \
        --weka prior-default:/weka/prior \
        --weka oe-training-default:/weka/oe-training-default \
        --priority high \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/ceres-cirrascale \
        --install "./scripts/install_molmo.sh && pip install -e .[eval]" \
        --allow-dirty \
        -- \
        python semantic_grasping/eval/eval_tg.py $TG_DIR $OUT_DIR $split $MODEL_ARGS --batch-size 8
done
