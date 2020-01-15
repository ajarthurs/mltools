"""Python multithreading.
"""

from threading import Thread

class ThreadWithReturnValueFromJoin(Thread):
  """XXX: Hack class to retrieve the target's return value at join.
  """
  def __init__(self, group=None, target=None, name=None,
               args=(), kwargs={}):
    Thread.__init__(self, group, target, name, args, kwargs)
    self._return = None

  def run(self):
    if self._target is not None:
      self._return = self._target(*self._args, **self._kwargs)

  def join(self, *args):
    Thread.join(self)
    return self._return


def create_thread(target_fn, target_args):
  return ThreadWithReturnValueFromJoin(
    target=target_fn,
    args=target_args,
    )


def run_thread_pool(threads):
  for thread in threads:
    thread.start()


def join_thread_pool(threads):
  return [thread.join() for thread in threads]
