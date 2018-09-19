#!/usr/bin/env sh

CONTAINER_SCRIPT_PATH=$1
export CLIPPER_MODEL_NAME=$2
IMPORT_ERROR_RETURN_CODE=3

export TF_CUDNN_USE_AUTOTUNE=0
export CLIPPER_MODEL_VERSION="1"
export CLIPPER_INPUT_TYPE="strings"
export CLIPPER_MODEL_PATH="/nexus-models-nsdi/store/tensorflow/ssd_mobilenet/v1_coco/"
export CLIPPER_IP="10.185.170.101" #IP address of the frontend container
export CLIPPER_PORT="7000" 
export GPU_ID=$3
echo $GPU_ID

/bin/bash -c "exec python $CONTAINER_SCRIPT_PATH"

if [ $? -eq $IMPORT_ERROR_RETURN_CODE ]; then
  echo "Encountered an ImportError when running container. You can use the pkgs_to_install argument when calling clipper_admin.build_model() to supply any needed Python packages."
  exit 1
fi

echo "Encountered error not related to missing packages. Please refer to the container log to diagnose."
exit 1
