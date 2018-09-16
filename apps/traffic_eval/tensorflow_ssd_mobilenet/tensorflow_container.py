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
import image_resize
from PIL import Image


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def tf_warmup(sess, num_images, x, o1, o2, o3):
    #NHWC_batch = np.zeros((num_images,300,300,3))
    #kk = skimage.io.imread("resized_images/image2.jpg")
    #NHWC_batch[0] = kk 
    #NHWC_batch = np.zeros((num_images,300,300,3))

    image = Image.open("images/image2.jpg")
    im_width, im_height = image.size

    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    results = sess.run([o1, o2, o3], feed_dict={x:image_np_expanded})

    boxes = results[1]
    ymin = int((boxes[0][0][0]*im_height))
    xmin = int((boxes[0][0][1]*im_width))
    ymax = int((boxes[0][0][2]*im_height))
    xmax = int((boxes[0][0][3]*im_width))

    Result = np.array(image_np[ymin:ymax,xmin:xmax])

    #(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
    #                              ymin * im_height, ymax * im_height)
    print(Result)
    
    return results

def predict_function(sess, imgs, x, y):
    NHWC_batch = np.zeros((len(imgs),300,300,3))

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
	self.input = graph.get_tensor_by_name('prefix/image_tensor:0')
	self.output1 = graph.get_tensor_by_name('prefix/num_detections:0')
	self.output2 = graph.get_tensor_by_name('prefix/detection_boxes:0')
	self.output3 = graph.get_tensor_by_name('prefix/detection_classes:0')

	self.sess = tf.Session(
		graph=graph,
		config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list=str(gpu_id))))

	#for i in range(129):
	#    tf_warmup(self.sess, i, self.input, self.output)
	a =  tf_warmup(self.sess, 1, self.input, self.output1, self.output2, self.output3)
	print(a)
	
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
    #try:
    #    model = TensorflowContainer(rpc_service.get_model_path(),
    #                            rpc_service.get_input_type(),
    #    			gpu_id)
    #    sys.stdout.flush()
    #    sys.stderr.flush()
    #except ImportError:
    #    sys.exit(IMPORT_ERROR_RETURN_CODE)
    #rpc_service.start(model)

	
    model = TensorflowContainer(rpc_service.get_model_path(),
                                rpc_service.get_input_type(),
        			2)
