if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

MODEL=$1

if [ "$MODEL" = "graspmolmo" ]; then
    MODEL_ARGS="eval_model=graspmolmo eval_model.ckpt_dir=/weka/oe-training-default/roseh/mm_olmo/robomolmo_checkpoints/graspmolmo_cotraining_06_evals_graspmolmo-focused_20250424_072247/step9000-unsharded"
elif [ "$MODEL" = "molmo" ]; then
    MODEL_ARGS="eval_model=molmo"
else
    echo "Unknown model: $MODEL"
    exit 1
fi

SPLITS="o t"
for split in $SPLITS; do
    gantry run -w ai2/abhayd -b ai2/prior \
        --name "tg_eval_${MODEL}_${split}" \
        --task-name "tg_eval_${MODEL}_${split}" \
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
        python semantic_grasping/eval/eval_tg.py split=$split $MODEL_ARGS
done
