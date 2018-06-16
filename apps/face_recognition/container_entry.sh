#!/usr/bin/env sh

CONTAINER_SCRIPT_PATH=$1
IMPORT_ERROR_RETURN_CODE=3

export CLIPPER_MODEL_NAME="vgg-face1"
export CLIPPER_MODEL_VERSION="1"
export CLIPPER_MODEL_PATH="/hdfs/msrlabs/v-haicsh/nexus-models-nsdi/store/caffe2/vgg_face/1"
export CLIPPER_IP="10.0.1.99" #IP address of the frontend container
export CLIPPER_PORT="7000" 
export GPU_ID=$2
echo $GPU_ID

/bin/bash -c "exec python $CONTAINER_SCRIPT_PATH"

if [ $? -eq $IMPORT_ERROR_RETURN_CODE ]; then
  echo "Encountered an ImportError when running container. You can use the pkgs_to_install argument when calling clipper_admin.build_model() to supply any needed Python packages."
  exit 1
fi

echo "Encountered error not related to missing packages. Please refer to the container log to diagnose."
exit 1
