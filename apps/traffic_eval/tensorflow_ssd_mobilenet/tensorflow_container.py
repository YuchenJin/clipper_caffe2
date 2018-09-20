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
import cv2
from PIL import Image


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print(im_width, im_height)
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def tf_warmup(sess, num_images, x, o1, o3):
    NHWC_batch = np.zeros((num_images,300,300,3))
    #curr_img = cv2.imread('resized_images/image1.jpg', 1)
    #retval, img_encoded = cv2.imencode('.jpg', curr_img)
    #jpg_as_text = base64.b64encode(img_encoded)
    #curr_img = base64.b64decode(jpg_as_text)
    #jpg_as_np = np.frombuffer(curr_img, dtype=np.uint8); 
    #curr_img = cv2.imdecode(jpg_as_np, flags=1)

    #image_np_expanded = np.expand_dims(curr_img, axis=0) 	
    #NHWC_batch[0] = image_np_expanded
    
    results = sess.run([o1, o3], feed_dict={x:NHWC_batch})
    return results

def predict_function(sess, imgs, x, o1, o3):
    NHWC_batch = np.zeros((len(imgs),300,300,3))

    for i, curr_img in enumerate(imgs):
        #curr_img = skimage.io.imread(StringIO(curr_img))
        #img = skimage.img_as_float(curr_img).astype(np.float32)
        #curr_img = cv2.imread((curr_img), 1)
        image_np_expanded = np.expand_dims(curr_img, axis=0) 	
        NHWC_batch[i] = image_np_expanded

    results = sess.run([o1, o3], feed_dict={x:NHWC_batch})
    results_str = []

    for i in range(len(imgs)):
        num_object = int(results[0][i])
        objects = [results[1][i][0:num_object]]
        cars = np.count_nonzero(objects == np.float32(3)) + np.count_nonzero(objects == np.float(8))
        people = np.count_nonzero(objects == np.float32(1))
    
    	results_str.append(','.join((str(cars), str(people))))
    return results_str

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
	self.input = graph.get_tensor_by_name('prefix/image_tensor:0')
	self.output1 = graph.get_tensor_by_name('prefix/num_detections:0')
	self.output2 = graph.get_tensor_by_name('prefix/detection_boxes:0')
	self.output3 = graph.get_tensor_by_name('prefix/detection_classes:0')

	self.sess = tf.Session(
		graph=graph,
		config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list=str(gpu_id))))

	for i in range(1,15):
	    tf_warmup(self.sess, i, self.input, self.output1, self.output3)
	    #tf_warmup(self.sess, i, self.input, self.output1, self.output2, self.output3)
	#print(a)
	
    def predict_strings(self, inputs):
        imgs = []
        for i in range(len(inputs)):
            img_bgr = base64.b64decode(inputs[i])
    	    jpg_as_np = np.frombuffer(img_bgr, dtype=np.uint8); 
    	    curr_img = cv2.imdecode(jpg_as_np, flags=1)
            imgs.append(curr_img)
        preds = self.predict_func(self.sess, imgs, self.input, self.output1, self.output3)
	
	result = []
	for i in range(len(preds)):
	    result.append(str(preds[i]))
   
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

	
    #model = TensorflowContainer(rpc_service.get_model_path(),
    #                            rpc_service.get_input_type(),
    #    			2)
