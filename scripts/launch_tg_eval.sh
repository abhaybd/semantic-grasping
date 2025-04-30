if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <model> [splits]"
    echo "  splits: space-separated list of splits (default: 'o t')"
    exit 1
fi

MODEL=$1
SPLITS=${2:-"o t"}
TIMESTAMP=$(date +%m%d_%H%M)

if [ "$MODEL" = "graspmolmo" ]; then
    MODEL_ARGS="eval_model=graspmolmo eval_model.ckpt_dir=/weka/oe-training-default/roseh/mm_olmo/robomolmo_checkpoints/graspmolmo_finetuning_ftmix_graspmolmo-ft_20250429_021138/step10000-unsharded"
elif [ "$MODEL" = "molmo" ]; then
    MODEL_ARGS="eval_model=molmo"
else
    echo "Unknown model: $MODEL"
    exit 1
fi

for split in $SPLITS; do
    gantry run -w ai2/molmo-act -b ai2/prior \
        --name "tg_eval_${MODEL}_${split}_${TIMESTAMP}" \
        --task-name "tg_eval_${MODEL}_${split}_${TIMESTAMP}" \
        --env-secret WANDB_API_KEY=ABHAYD_WANDB_API_KEY \
        --env-secret GITHUB_TOKEN=ABHAYD_GITHUB_TOKEN \
        --gh-token-secret ABHAYD_GH_TOKEN \
        --dataset-secret ABHAYD_SSH_KEY:/root/.ssh/id_ed25519 \
        --beaker-image ai2/cuda11.8-dev-ubuntu20.04 \
        --gpus 1 \
        --weka prior-default:/weka/prior \
        --weka oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/ceres-cirrascale \
        --install "./scripts/install_molmo.sh && pip install -e .[eval]" \
        --allow-dirty \
        -- \
        python semantic_grasping/eval/eval_tg.py split=$split $MODEL_ARGS "+fold=\"0\""
done
