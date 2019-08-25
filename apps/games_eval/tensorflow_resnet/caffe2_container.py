from __future__ import print_function
import rpc
import os
import sys
import glob
import caffe2
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
import base64
import time
from StringIO import StringIO


def warmup(net_def, n, device_opts):
    NCHW_batch = np.zeros((n,3,224,224))
    workspace.FeedBlob('data', NCHW_batch.astype(np.float32), device_opts)
    start = time.time()
    workspace.RunNet(net_def.name, 1)
    end = time.time()
    print(end-start)

def predict_function(net_def, device_opts, imgs):
    NCHW_batch = np.zeros((len(imgs),3,224,224))

    for i, curr_img in enumerate(imgs):
	curr_img = skimage.io.imread(StringIO(curr_img))
        img = skimage.img_as_float(curr_img).astype(np.float32)
        img = img.swapaxes(1, 2).swapaxes(0, 1)
        img = img[(2, 1, 0), :, :]
        img = img * 255 - 128
        NCHW_batch[i] = img

    workspace.FeedBlob('data', NCHW_batch.astype(np.float32), device_opts)
    workspace.RunNet(net_def.name, 1)
    results = workspace.FetchBlob('prob')
    return results


class caffe2Container(rpc.ModelContainerBase):
    def __init__(self, path, input_type, gpu_id):
        self.device_opts = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
	print(self.device_opts)

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

	warmup(self.net_def, 1, self.device_opts) 

    def predict_strings(self, inputs):
        imgs = []
        for i in range(len(inputs)):
            img_bgr = base64.b64decode(inputs[i])
            imgs.append(img_bgr)
        preds = self.predict_func(self.net_def, self.device_opts, imgs)
	
	result = []
	for i in range(len(preds)):
	    #result.append(preds[i])
	    #result.append(preds[i].index(max(preds[i])))
	    result.append(str(np.argmax(preds[i])))
   
	return result
    
    
if __name__ == "__main__":
    print("Starting Caffe2 container")

    if "GPU_ID" in os.environ:
        gpu_id = int(os.environ["GPU_ID"])
    else:
        print("GPU id not set")

    print("Init model")

    rpc_service = rpc.RPCService()
    try:
        model = caffe2Container(rpc_service.get_model_path(),
                                rpc_service.get_input_type(),
				gpu_id)
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)
    rpc_service.start(model)

