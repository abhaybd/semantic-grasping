beaker session create -w ai2/abhayd --budget ai2/prior \
    --name "molmo_tmp_inference_server" \
    --mount src=weka,ref=prior-default,dst=/weka/prior \
    --mount src=weka,ref=oe-training-default,dst=/weka/oe-training-default \
    --workdir /weka/prior/abhayd/semantic-grasping \
    --detach \
    --bare \
    --image beaker://ai2/cuda11.8-dev-ubuntu20.04 \
    --priority high \
    --gpus 1 \
    --cluster ai2/neptune-cirrascale \
    --port 8080:8080 \
    -- \
    /weka/prior/abhayd/envs/grasping/bin/fastapi run semantic_grasping/eval/inference_server.py --port 8080
