from __future__ import print_function
import rpc
import os
import sys
import glob
import caffe2
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import numpy as np


def predict_function(net_def, device_opts, inputs):
    workspace.FeedBlob('data', np.random.rand(1, 3, 227, 227).astype(np.float32), device_opts)
    workspace.RunNet(net_def.name, 1)

class caffe2Container(rpc.ModelContainerBase):
    def __init__(self, path, input_type, gpu_id):
        self.device_opts = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)

        self.input_type = rpc.string_to_input_type(input_type)
        modules_folder_path = "{dir}/".format(dir=path)
        INIT_NET = "{dir}/init_net.pb".format(dir=modules_folder_path)
        PREDICT_NET = "{dir}/predict_net.pb".format(dir=modules_folder_path)
        self.predict_func = predict_function

        init_def = caffe2_pb2.NetDef()
        with open(INIT_NET, 'r') as f:
            init_def.ParseFromString(f.read())
            init_def.device_option.CopyFrom(self.device_opts)
            workspace.RunNetOnce(init_def)

        self.net_def = caffe2_pb2.NetDef()
        with open(PREDICT_NET, 'r') as f:
            self.net_def.ParseFromString(f.read())
            self.net_def.device_option.CopyFrom(self.device_opts)
            workspace.CreateNet(self.net_def, overwrite=True)


    def predict_strings(self, inputs):
        preds = self.predict_func(self.net_def, self.device_opts, inputs)
        return [str("OK")]

    
if __name__ == "__main__":
    print("Starting Caffe2 container")
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    #ip = "localhost"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
        print("Connecting to Clipper on {ip}").format(ip=ip)
    else:
        print(
            "ERROR: CLIPPER_IP environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "strings"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]
    else:
        print("Using default input type: strings")

    if "CLIPPER_MODEL_PATH" in os.environ:
        path = str(os.environ["CLIPPER_MODEL_PATH"])
    else:
        print("Clipper model path not found.")

    if "GPU_ID" in os.environ:
        gpu_id = int(os.environ["GPU_ID"])
    else:
        print("GPU id not set")

    model = caffe2Container(path, input_type, gpu_id)
    #model = SumContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, port, model_name, model_version, input_type)
