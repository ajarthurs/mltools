"""Logging configuration.
"""

import logging
import os
import sys
from pythonjsonlogger import jsonlogger

def create_log_config(log_path=None):
  if not log_path:
    this_path = os.path.realpath(__file__)
    log_path = os.path.dirname(this_path)
  try:
    os.mkdir(log_path)
  except FileExistsError:
      pass
  config = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
      'primary': {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
      },
      'simple': {
        'format': '%(message)s',
      },
      'json': {
        'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
      },
    },

    'handlers': {
      'consoleHandler': {
        'class': 'logging.StreamHandler',
        'level': 'INFO',
        'formatter': 'primary',
        'stream': 'ext://sys.stdout',
      },
      'jsonFileHandler': {
        'class': 'logging.FileHandler',
        'level': 'DEBUG',
        'formatter': 'json',
        'filename': os.path.join(log_path, 'mltools-log.json'),
      },
    },

    'loggers': {
      '': { # root logger
        'handlers': [
          'consoleHandler',
          'jsonFileHandler',
        ],
        'level': 'DEBUG',
      },
    },
  }
  return config
