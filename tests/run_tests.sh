#!/bin/bash

set -e # Exit immediately if any command fails

export COMPOSE_PROJECT_NAME=docker-training-pytorch

echo "🏗 Building all images"
docker compose build

# Step 1: Run the docker compose services
echo "🚀 Running the docker compose services..."
docker compose run train
echo "✅  Train Run Successfully. ✅ "
docker compose run eval
echo "✅  Evaluation Run Successfully. ✅ "
docker compose run infer
echo "✅ Inference  finished  Successfully. ✅  "

# Step 2: Check if the checkpoint is saved in the volume
echo "🔍 Checking for checkpoint file..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_docker_vol:/opt/mount busybox ls /opt/mount/model/model_checkpoint_path.pth > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Checkpoint file found."
else
    echo "❌ Checkpoint file not found!"
    exit 1
fi

# Step 3: Check if the eval.json output is saved in the volume
echo "🔍 Checking for eval_results.json file..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_docker_vol:/opt/mount busybox  ls /opt/mount/model/eval_results.json > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ eval_results.json file found."
else
    echo "❌ eval_results.json file not found!"
    exit 1
fi

# Step 4: Print the output of eval_results.json
echo "📄 Printing the content of eval_results.json file..."
docker run --rm -v ${COMPOSE_PROJECT_NAME}_docker_vol:/opt/mount busybox cat /opt/mount/model/eval_results.json 


# Step 5: Check if inference results are saved in the volume
echo "🔍 Checking for inference results..."
inference_count=$(docker run --rm -v ${COMPOSE_PROJECT_NAME}_docker_vol:/opt/mount busybox sh -c "ls /opt/mount/responses/results/*.png | wc -l")
if [ "$inference_count" -eq 5 ]; then
    echo "✅ 5 inference result images found."
else
    echo "❌ Expected 5 inference result images, but found $inference_count!"
    exit 1
fi
