"""Plot metrics from one or more data series.
"""

import argparse
import logging
import logging.config
import os

log = logging.getLogger(__name__)

def parse_args(argv=None):
  import argparse
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
    '-t', '--title',
    help='Name of chart.',
    required=True,
    )

  parser.add_argument(
    '-x', '--variable_name',
    help='Name of independent variable used in all data series.',
    required=True,
    )

  parser.add_argument(
    '-y', '--function_name',
    help='Name of dependent variable used in all data series.',
    required=True,
    )

  parser.add_argument(
    '-n',  '--series_names',
    help='One or more names of data series.',
    metavar='NAME',
    nargs='+',
    )

  parser.add_argument(
    '--histogram',
    help='Plot as a histogram instead. Default is to plot as a series.',
    action='store_true',
    )

  parser.add_argument(
    '-i', '--metrics_pickle_paths',
    help='One or more paths to pickle files containing the data series.',
    metavar='PATH',
    nargs='+',
    )

  parser.add_argument(
    '-o', '--plot_output_path',
    help='Path to plot output file.',
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
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import pickle
  recorded_metrics_list = []
  i = 0
  for path in args.metrics_pickle_paths:
    recorded_script, recorded_args, recorded_metrics = pickle.load(
      open(path, 'rb')
      )
    recorded_metrics_list.append(recorded_metrics)
    i += 1
  log.debug(recorded_metrics_list)
  generate_multiseries_plot(
    plt=plt,
    histogram=args.histogram,
    title=args.title,
    variable_name=args.variable_name,
    function_name=args.function_name,
    series_names=args.series_names,
    metrics_list=recorded_metrics_list,
    )
  plt.savefig(args.plot_output_path, dpi=300)
  return 0


def generate_multiseries_plot(plt, histogram, title, variable_name, function_name, series_names, metrics_list):
  from matplotlib import gridspec
  plt.figure(1)
  gs = gridspec.GridSpec(1, 1)
  series_axes1 = plt.subplot(gs[0, 0])
  i = 0
  for metrics in metrics_list:
    function = [batch[function_name] for batch in metrics]
    if histogram:
      series_axes1.hist(
        function,
        bins='auto',
        label=series_names[i],
        alpha=0.75,
        )
    else:
      series_axes1.plot(
        function,
        label=series_names[i],
        )
    i += 1
  plt.title(title)
  if histogram:
    plt.ylabel(variable_name)
    plt.xlabel(function_name)
  else:
    plt.xlabel(variable_name)
    plt.ylabel(function_name)
  plt.legend()


if __name__ == '__main__':
  args = parse_args()
  run(args)
