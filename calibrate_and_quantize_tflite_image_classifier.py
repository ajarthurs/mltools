"""Generate calibration data-annotated TFLite model.

Requires Tensorflow 1.13+.
"""

import logging
import logging.config
import os
from mltools.cv2_preprocessors import preprocessors

log = logging.getLogger(__name__)

def parse_args(argv=None):
  import argparse

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    '-i', '--input_tflite_model_path',
    help='Path to image classifier TFLite model.',
    required=True,
    )

  parser.add_argument(
    '-o', '--output_tflite_model_path',
    help='Path to image classifier TFLite model.',
    required=True,
    )

  parser.add_argument(
    '-p', '--model_preprocessor',
    help='Model preprocessor.',
    choices=preprocessors().keys(),
    required=True,
    )

  parser.add_argument(
    '--delegate_to_tpu',
    help='Delegate inference to TPU.',
    type=bool,
    default=False,
    )

  parser.add_argument(
    '-l', '--image_file_list_path',
    help='Path to image classifier TFLite model.',
    required=True,
    )

  parser.add_argument(
    '-d', '--dataset_split_path',
    help='Path to directory containing a dataset split.',
    required=True,
    )

  parser.add_argument(
    '--log_path',
    help='Path to log directory. Default is same directory as this script.',
    )

  if argv:
    args = parser.parse_args(argv)
  else:
    args = parser.parse_args()

  from mltools.python_logging import create_log_config
  log_config = create_log_config(
    log_path=args.log_path,
    )
  logging.config.dictConfig(log_config)
  log = logging.getLogger(__name__)
  log.info('%s %s' % (os.path.basename(__file__), args))
  return args


def run(args):
  """Calibrate and quantize model.
  """
  import numpy as np
  import tensorflow as tf
  from tensorflow.lite.python.optimize import calibrator

  image_file_list = None
  with open(args.image_file_list_path) as f:
    image_file_list = f.read().splitlines()
  model_preprocessor_fn=preprocessors()[args.model_preprocessor]

  def _input_gen():
    """Setup a graph to yield one batch of images.
        Returns:
          [image] generator: A batch of N images, each image shape being exactly [1, model_input_size, model_input_size, num_channels].
    """
    import cv2
    i=0
    for image_file in image_file_list:
      log.info('Image %d of %d: %s' % (
        i,
        len(image_file_list) - 1,
        image_file,
        ))
      cv2_image = cv2.imread(os.path.join(args.dataset_split_path, image_file))
      preprocessed_image = model_preprocessor_fn(cv2_image)
      image = np.array(preprocessed_image)
      yield [[image]]
      i+=1

  float32_tflite_model = open(args.input_tflite_model_path, 'rb').read()
  quantizer = calibrator.Calibrator(float32_tflite_model)
  tflite_quantized_model = quantizer.calibrate_and_quantize(
      _input_gen,
      allow_float=False,
      input_type=tf.int8,
      output_type=tf.int8,
      )
  with open(args.output_tflite_model_path, 'wb') as outfile:
    outfile.write(tflite_quantized_model)


if __name__ == '__main__':
  args = parse_args()
  run(args)
