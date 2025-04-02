beaker session create -w ai2/abhayd --budget ai2/prior \
    --gpus 4 \
    --shared-memory 128GiB \
    --secret-env WANDB_API_KEY=WANDB_API_KEY \
    --mount src=beaker,ref=abhayd/dataset_0324_2125,dst=/train_data/dataset \
    --mount src=beaker,ref=abhayd/obsgen_0324_1943,dst=/train_data/data \
    --bare
