import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""

        # self.writer = tf.summary.FileWriter(log_dir)  # for stargan_env
        self.writer = tf.compat.v1.summary.FileWriter(log_dir) # for ld

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""

        # for stargan_env
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)

        # for ld
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)