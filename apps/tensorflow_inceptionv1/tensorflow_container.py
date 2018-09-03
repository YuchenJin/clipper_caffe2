from __future__ import print_function
import rpc
import os
import sys
import glob
import numpy as np
import skimage.io
import skimage.transform
import base64
from StringIO import StringIO
import json
import tensorflow as tf


def predict_function(sess, imgs, x, y):
    NHWC_batch = np.zeros((len(imgs),224,224,3))

    for i, curr_img in enumerate(imgs):
        curr_img = skimage.io.imread(StringIO(curr_img))
        img = skimage.img_as_float(curr_img).astype(np.float32)
        NHWC_batch[i] = img

    results = sess.run(y, feed_dict={x:NHWC_batch})
    return results

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

class TensorflowContainer(rpc.ModelContainerBase):
    def __init__(self, path, input_type, gpu_id):
        self.input_type = rpc.string_to_input_type(input_type)
        frozen_model_path = glob.glob(os.path.join(path, "*.pb"))[0]
        self.predict_func = predict_function
	graph = load_graph(frozen_model_path)

	# Access the input and output nodes 
	self.input = graph.get_tensor_by_name('prefix/input:0')
	self.output = graph.get_tensor_by_name('prefix/output:0')

	self.sess = tf.Session(
		graph=graph,
		config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list=str(gpu_id))))

    def predict_strings(self, inputs):
        imgs = []
        for i in range(len(inputs)):
            img_bgr = base64.b64decode(inputs[i])
            imgs.append(img_bgr)
        preds = self.predict_func(self.sess, imgs, self.input, self.output)
	
	result = []
	for i in range(len(preds)):
	    result.append(str(np.argmax(preds[i])))
   
	return result
    
    
if __name__ == "__main__":
    print("Starting Tensorflow container")

    if "GPU_ID" in os.environ:
        gpu_id = int(os.environ["GPU_ID"])
    else:
        print("GPU id not set")

    print("Init model")

    rpc_service = rpc.RPCService()
    try:
        model = TensorflowContainer(rpc_service.get_model_path(),
                                rpc_service.get_input_type(),
        			gpu_id)
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)
    rpc_service.start(model)
