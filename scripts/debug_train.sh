beaker session create -w ai2/abhayd --budget ai2/prior \
    --gpus 4 \
    --shared-memory 64GiB \
    --mount src=beaker,ref=abhayd/dataset_test_0314_1317,dst=/train_data/dataset \
    --mount src=beaker,ref=abhayd/obsgen_0311_1420,dst=/train_data/data \
    --bare
