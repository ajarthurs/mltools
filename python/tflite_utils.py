"""Utility functions for TFLite.
"""

import logging
import math
import numpy as np
import os
import tensorflow.lite as tflite

log = logging.getLogger(__name__)

def evaluate_image(interpreter, image):
  """Call the TFLite interpreter to evaluate an image.

  Args:
    interpreter: Allocated TFLite interpreter.
    image: NumPy image of shape (height, width, channels).

  Returns:
    predictions: NumPy array of shape (classes).
  """
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']
  image = image.reshape(input_shape)
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  predictions = interpreter.get_tensor(output_details[0]['index'])
  return np.squeeze(predictions)


def create_interpreter_pool(size, tflite_path=None, tflite_model=None, delegate_to_tpu=False):
  """Create a pool of TFLite interpreters.
    Args:
      size: Target size of pool.
      tflite_path: File path to TFLite model. Takes precedence over tflite_model if specified.
      tflite_model: Preloaded TFLite model.
      delegate_to_tpu: Delegate inference to TPU.

    Returns:
      interpreters: List of TFLite interpreters.

    Throws:
      Invalid TFLite model.
  """
  if tflite_path:
    tflite_model = open(tflite_path, 'rb').read()
  if not tflite_model:
    raise ValueError('Invalid TFLite model. Check either tflite_path (%s) or tflite_model.' % tflite_path)
  if delegate_to_tpu:
    from tensorflow.lite.python.interpreter import load_delegate
    interpreters = [
      tflite.Interpreter(
        model_content=tflite_model,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')],
        ) for i in range(size)
      ]
  else:
    interpreters = [
      tflite.Interpreter(
        model_content=tflite_model,
        ) for i in range(size)
      ]
  for interpreter in interpreters:
    interpreter.allocate_tensors()
  input_details = interpreters[0].get_input_details()
  output_details = interpreters[0].get_output_details()
  log.debug({
    'Input info': input_details,
    'Output info': output_details,
    })
  return interpreters


def create_dataset_split_batch_queue(dataset_split_path, dataset_split_map_file, batch_size, model_input_details, model_preprocessor_fn, model_labels_offset, read_from_numpy=False, max_samples=None, quantize_input=True):
  """Create a batch queue that preprocesses a dataset for a given model.

  Args:
    dataset_split_path: Path to dataset split's directory.
    dataset_split_map_file: Name of dataset split's map file.
    batch_size: Number of images per batch. Default is the entire dataset split.
    model_input_details: TFLite model input details (see tflite.Interpreter.get_input_details()). Not used if `read_from_numpy` is True.
    model_preprocessor_fn: Function that preprocesses each image to fit the model. Not used if `read_from_numpy` is True.
    model_labels_offset: Label offset.
    read_from_numpy: Return a callback that loads preprocessed images from NumPy files. Default is preprocess images on-the-fly.
    max_samples: Maximum samples to include. Default is entire split.
    quantize_input: Quantize images during preprocessing. Only applies to quantized models.

  Returns:
    dataset: Dictionary of dataset split details.
    read_batch_fn: Batch generator function.
      Signature: (batch_id)
        Args:
          batch_id: Batch index.
        Returns:
          List of images at batch index.
  """
  dataset = {}
  dataset_valmap = os.path.join(dataset_split_path, dataset_split_map_file) # FILENAME GT_LABEL_ID
  with open(dataset_valmap, 'r') as f:
    image_maps = [l.split() for l in f.readlines()]
  dataset['files'] = [os.path.join(dataset_split_path, m[0]) for m in image_maps]
  dataset['gtlabels'] = [int(m[1]) for m in image_maps]
  if max_samples:
    dataset['num_samples'] = min(max_samples, len(image_maps))
  else:
    dataset['num_samples'] = len(image_maps)
  dataset['num_batches'] = int(math.ceil(dataset['num_samples'] / float(batch_size)))

  def _maybe_quantize(image):
    input_dtype = model_input_details[0]['dtype']
    if quantize_input and (input_dtype == np.int8 or input_dtype == np.uint8):
      input_quant = model_input_details[0]['quantization']
      input_scale = input_quant[0]
      input_zp = input_quant[1]
      quant_image = (np.round(image/input_scale) + input_zp).astype(input_dtype)
    else:
      quant_image = image
    return quant_image

  def _numpy_read_batch_fn(batch_id):
    batch_index = batch_id * batch_size
    batch_files = dataset['files'][batch_index:batch_index + batch_size]
    i = 0
    for image_file in batch_files:
      image = np.load('%s.npy' % image_file)
      image = _maybe_quantize(image)
      yield [image, dataset['gtlabels'][batch_index + i] + model_labels_offset]
      i += 1

  def _cv2_read_batch_fn(batch_id):
    import cv2
    batch_index = batch_id * batch_size
    batch_files = dataset['files'][batch_index:batch_index + batch_size]
    i = 0
    for image_file in batch_files:
      cv2_image = cv2.imread(image_file)
      preprocessed_image = model_preprocessor_fn(cv2_image)
      image = np.array(preprocessed_image)
      image = _maybe_quantize(image)
      yield [image, dataset['gtlabels'][batch_index + i] + model_labels_offset]
      i += 1

  if read_from_numpy:
    read_batch_fn = _numpy_read_batch_fn
  else:
    read_batch_fn = _cv2_read_batch_fn
  return dataset, read_batch_fn


def create_image_classification_thread_pool(interpreters, image_batch, create_thread_fn):
  """Create a pool to evaluate a batch of images.
    Args:
      interpreters: List of allocated TFLite interpreters (see create_interpreter_pool).
      image_batch: NumPy array of NumPy images of shape (batch_size, height, width, channels).
      create_thread_fn: Function that creates a thread.
        Signature: (target_fn, target_args)
          Args:
            target_fn: Target function.
            target_args: Tuple of arguments for the target function.
          Returns:
            Thread object.
  """
  size = len(image_batch)
  return [
    create_thread_fn(
      target_fn=evaluate_image,
      target_args=(interpreters[i], image_batch[i]),
      )
    for i in range(size)
    ]
