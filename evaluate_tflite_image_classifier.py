"""Evaluate TFLite Image Classifier.

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
    '-i', '--tflite_model_path',
    help='Path to image classifier TFLite model.',
    required=True,
    )

  parser.add_argument(
    '-p', '--model_preprocessor',
    help='Model preprocessor.',
    choices=preprocessors.keys(),
    required=True,
    )

  parser.add_argument(
    '-l', '--model_labels_offset',
    help='An offset for the labels in the dataset. This flag is primarily used to evaluate architectures such as ResNet that do not use a background class.',
    type=int,
    default=0,
    )

  parser.add_argument(
    '-d', '--dataset_valsplit_path',
    help='Path to directory containing the dataset validation split.',
    required=True,
    )

  parser.add_argument(
    '-s', '--max_samples',
    help='Maximum number of samples to process. Default is all samples.',
    type=int,
    )

  parser.add_argument(
    '-b', '--batch_size',
    help='The number of samples in each batch. Default is number of CPUs.',
    default=os.cpu_count(),
    )

  parser.add_argument(
    '--log_path',
    help='Path to log directory. Default is same directory as this script.',
    )

  parser.add_argument(
    '--mlperf_compat_output',
    help='Path to MLPerf-compatible output file that will store results.',
    )

  if argv:
    args = parser.parse_args(argv)
  else:
    args = parser.parse_args()

  from python_logging import create_log_config
  log_config = create_log_config(
    log_path=args.log_path,
    )
  logging.config.dictConfig(log_config)
  log = logging.getLogger(__name__)
  log.info('%s %s' % (os.path.basename(__file__), args))
  return args

def run(args):
  """Infer dataset's validation images with a TFLite image classifier.
  """
  import pickle
  import numpy as np

  from python_threading import create_thread, run_thread_pool, join_thread_pool
  from tflite_utils import create_dataset_split_batch_queue, create_interpreter_pool, create_image_classification_thread_pool

  interpreters = create_interpreter_pool(
    tflite_path=args.tflite_model_path,
    size=args.batch_size,
    )
  dataset_batch_queue, read_batch_fn = create_dataset_split_batch_queue(
    dataset_split_path=args.dataset_valsplit_path,
    dataset_split_map_file='val_map.txt',
    batch_size=args.batch_size,
    max_samples=args.max_samples,
    model_input_details=interpreters[0].get_input_details(),
    model_preprocessor_fn=preprocessors[args.model_preprocessor],
    model_labels_offset=args.model_labels_offset,
    )
  total_true_positives = 0
  total_top_5_true_positives = 0
  total_images = 0
  metrics = [{}]*dataset_batch_queue['num_batches']
  if args.mlperf_compat_output:
    mlperf_log = open(args.mlperf_compat_output, 'w')
  log.info('Processing %d batches of size %d...' % (dataset_batch_queue['num_batches'], args.batch_size))
  for batch_id in range(dataset_batch_queue['num_batches']):
    image_batch, gtlabel_batch = zip(*[image_map for image_map in read_batch_fn(batch_id)])
    num_images = len(image_batch)
    threads = create_image_classification_thread_pool(
      interpreters=interpreters,
      image_batch=image_batch,
      create_thread_fn=create_thread,
      )
    run_thread_pool(threads)
    predictions = np.array(join_thread_pool(threads))
    top_5_labels = [prediction.argsort()[-5:] for prediction in predictions]
    log.debug({'batch_top_5_labels': top_5_labels})
    batch_true_positives = np.sum(
      [int(top_5_labels[i][-1] == gtlabel_batch[i]) for i in range(num_images)],
      )
    batch_top_5_true_positives = np.sum(
      [int(gtlabel_batch[i] in top_5_labels[i]) for i in range(num_images)],
      )
    batch_accuracy = batch_true_positives / num_images
    batch_top_5_accuracy = batch_top_5_true_positives / num_images
    if args.mlperf_compat_output:
      tp = total_true_positives
      ti = total_images
      for i in range(num_images):
        tp += int(top_5_labels[i][-1] == gtlabel_batch[i])
        ti += 1
        mlperf_log.write('self.good = %d/%d = %.3f, result = %d, expected = %d\n' % (
          batch_id*args.batch_size + i,
          dataset_batch_queue['num_samples'] - 1,
          tp / ti,
          top_5_labels[i][-1],
          gtlabel_batch[i],
        ))
    total_true_positives += batch_true_positives
    total_top_5_true_positives += batch_top_5_true_positives
    total_images += num_images
    cumulative_accuracy = total_true_positives / total_images
    cumulative_top_5_accuracy = total_top_5_true_positives / total_images
    metrics[batch_id]['batch_accuracy'] = {
      'gtlabels': gtlabel_batch,
      'top_5_labels': top_5_labels,
      'accuracy': batch_accuracy,
      'top_5_accuracy': batch_top_5_accuracy,
      'cumulative_accuracy': cumulative_accuracy,
      'cumulative_top_5_accuracy': cumulative_top_5_accuracy,
      }
    log.debug(metrics[batch_id]['batch_accuracy'])
    log.info('Batch %d of %d: %0.2f%%' % (
      batch_id,
      dataset_batch_queue['num_batches'] - 1,
      100*metrics[batch_id]['batch_accuracy']['cumulative_accuracy'],
      ))
  with open(os.path.join(args.log_path, 'metrics.pickle'), 'wb') as f:
    pickle.dump(metrics, f)
  return 0

if __name__ == '__main__':
  args = parse_args()
  run(args)
