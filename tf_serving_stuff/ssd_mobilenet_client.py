# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import numpy as np
import cv2
import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image','./images/image2.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  image = cv2.imread(FLAGS.image, 1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (300, 300))
  image_np_expanded = np.expand_dims(image, axis=0)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'ssd'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image_np_expanded, shape=[1, 300, 300, 3]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()
