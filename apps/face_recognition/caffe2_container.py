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
from StringIO import StringIO


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled

def predict_function(net_def, device_opts, imgs):
    NCHW_batch = np.zeros((len(imgs),3,224,224))

    for i, curr_img in enumerate(imgs):
	curr_img = skimage.io.imread(StringIO(curr_img))
        img = skimage.img_as_float(curr_img).astype(np.float32)
        img = rescale(img, 224, 224)
        img = crop_center(img, 224, 224)
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

        self.input_type = rpc.string_to_input_type(input_type)
        modules_folder_path = "{dir}/".format(dir=path)
        INIT_NET = "{dir}/init_net.pb".format(dir=modules_folder_path)
        PREDICT_NET = "{dir}/predict_net.pb".format(dir=modules_folder_path)
	
	with open('/hdfs/pnrsy/v-haicsh/nexus-models/store/caffe2/vgg_face/names.txt', 'rb') as fd:
	     self.cns = [l.rstrip() for l in fd]
	
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
        imgs = []
        for i in range(len(inputs)):
            img_bgr = base64.b64decode(inputs[i])
            imgs.append(img_bgr)
        preds = self.predict_func(self.net_def, self.device_opts, imgs)
	
	result = []
	for i in range(len(preds)):
            name = self.cns[np.argmax(preds[i])]
	    result.append(name)
   
	return result
    
    #def predict_bytes(self, inputs):
    #    imgs = []
    #    for i in range(len(inputs)):
    #        img_bgr = cv2.imdecode(inputs[i], cv2.CV_LOAD_IMAGE_COLOR)
    #        imgs.append(img_bgr)

    #    preds = self.predict_func(self.net_def, self.device_opts, imgs)

    
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

    print("Init model")
    model = caffe2Container(path, input_type, gpu_id)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, port, model_name, model_version, input_type)
