#!/usr/bin/env python
"""
File: model_trainer.py
Date: 11/17/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import logging, argparse
import datetime

import tensorflow as tf

from deep_model.config import Configuration
from deep_model.params import Params
from deep_model.ops import f1

class ModelTrainer(object):

    def __init__(self, model, config, params, logger, restore_model_path=None):
        assert isinstance(logger, logging.Logger)
        assert isinstance(config, Configuration)
        assert isinstance(params, Params)

        self.model = model
        self.config = config
        self.params = params
        self.logger = logger

        self.restore = restore_model_path is not None
        self.restore_model_path = restore_model_path

        self.tensorboard_dir = os.path.join(config.tensorboard_dir, self._get_job_name())

        self.logging_metrics = dict()
        self.tensorboard_metrics = dict()

        tf.random.set_random_seed(params.seed)

        self.epoch = 0

    def train(self, train_dataset, test_dataset, trainable_scopes=None):
        assert isinstance(train_dataset, tf.data.Dataset)
        assert isinstance(test_dataset, tf.data.Dataset)

        self._setup_dataset_iterators(train_dataset, test_dataset)

        input, latent_priors, labels = self.iterator.get_next()
        # input, labels = self.iterator.get_next()
        input = tf.identity(input, "input")  # name the input tensor

        # Create the model's computation graph
        self.logger.info("Instantiating model...")

        self.is_training = tf.placeholder(tf.bool)
        self.output, self.cost = self.model(input, latent_priors, labels, self.is_training)
        self.output = tf.identity(self.output, "output")

        self._define_logging_metrics(self.output, labels)

        # Get list of variables to train
        var_list = None
        if trainable_scopes is not None:
            self.logger.info("Trainable scopes: %s" % trainable_scopes)
            var_list = list()
            for scope in trainable_scopes:
                vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                var_list.extend(vars)

        # Define the optimization strategy
        self.optimizer, self.global_step = self._get_optimizer(self.cost, var_list=var_list)

        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        self.logger.debug("Creating TensorFlow session")
        with tf.Session() as self.sess:
            self._configure_tensorboard()

            # Initialize graph, data iterators, and model saver
            self.sess.run(init)
            self.sess.run(init_l)

            self.train_handle = self.sess.run(self.train_iterator.string_handle())
            self.saver = self._make_saver()
            self.saver.save(self.sess, self.config.model_file, global_step=self.global_step)

            if self.restore:
                self.logger.info("Restoring model from checkpoint: %s" % self.restore_model_path)
                saver = tf.train.Saver(self.model.restore_tensors)
                saver.restore(self.sess, tf.train.latest_checkpoint(self.restore_model_path))
                self.logger.info("Model restored.")

            # Training epochs
            self.logger.info("Training...")
            for self.epoch in range(self.params.epochs):
                self.sess.run(self.train_iterator.initializer)

                self.batch = 0
                while True:
                    try:
                        self._train_batch(self.train_handle)

                        if self.batch % self.config.tensorboard_freq == 0:
                            self.sess.run(self.test_iterator.initializer)
                            self.test_handle = self.sess.run(self.test_iterator.string_handle())

                            self._report_batch()
                            self._log_tensorboard()

                        if self.batch % self.config.save_freq == 0:
                            self._save_model()

                    except tf.errors.OutOfRangeError:
                        self.logger.info("End of epoch %d" % self.epoch)
                        break

            self.logger.info("Training complete.")
            self._save_model()

    def _make_saver(self):
        vars_to_save = tf.trainable_variables() + self.model.variables_to_save
        return tf.train.Saver(vars_to_save, save_relative_paths=True,
                              max_to_keep=self.config.max_to_keep,
                              keep_checkpoint_every_n_hours=self.config.keep_checkpoint_every_n_hours)

    def _define_logging_metrics(self, output, labels):

        predictions = tf.round(output)
        correct = tf.equal(predictions, tf.round(labels))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        positive_mask = tf.equal(tf.round(labels), 1)
        correct_positive = tf.boolean_mask(correct, positive_mask)
        positive_accuracy = tf.reduce_mean(tf.cast(correct_positive, tf.float32))

        self.logging_metrics["cost"] = self.cost
        self.logging_metrics["F1"] = f1(labels, predictions)
        self.logging_metrics["accuracy"] = accuracy
        self.logging_metrics["positive accuracy"] = positive_accuracy

    def _setup_dataset_iterators(self, train_dataset, test_dataset):
        assert isinstance(train_dataset, tf.data.Dataset)
        assert isinstance(test_dataset, tf.data.Dataset)

        # dataset iterators (for selecting dataset to feed in)
        self.dataset_handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle,
                                                            train_dataset.output_types,
                                                            train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()

    def _configure_tensorboard(self):

        # Configure all of the metrics to log to TensorBoard
        metrics = list()

        train_cost = tf.summary.scalar('train_cost', self.cost)
        metrics.append(train_cost)

        # Also add all of the logging metrics
        for metric_name in self.logging_metrics:
            metric_tensor = self.logging_metrics[metric_name]
            metric_summary = tf.summary.scalar("test_%s" % metric_name, metric_tensor)
            metrics.append(metric_summary)

        self.merged_summary = tf.summary.merge(metrics)
        self.writer = tf.summary.FileWriter(logdir=self.tensorboard_dir)

        # Add the pretty graph viz
        self.writer.add_graph(self.sess.graph)

    def _train_batch(self, train_set_handle):
        feed_dict = {self.is_training: True,
                     self.dataset_handle: train_set_handle}

        train_summary, _, cost = self.sess.run([self.merged_summary, self.optimizer, self.cost],
                                               feed_dict=feed_dict)

        self.logger.info("Epoch: %d, Batch %d: cost: %f" % (self.epoch, self.batch, cost))
        self.writer.add_summary(train_summary, global_step=self.sess.run(self.global_step))
        self.batch += 1

    def _report_batch(self):
        for metric_name in self.logging_metrics:
            tensor = self.logging_metrics[metric_name]
            value = self.sess.run(tensor, feed_dict={self.is_training: False,
                                                     self.dataset_handle: self.test_handle})

            self.logger.info("Test %s: %s" % (metric_name, value))

    def _log_tensorboard(self):
        self.logger.info("Logging test output to TensorBoard")

        test_summary = self.sess.run(self.merged_summary,
                                     feed_dict={self.is_training: False,
                                                self.dataset_handle: self.test_handle})

        self.writer.add_summary(test_summary, global_step=self.sess.run(self.global_step))
        self.writer.flush()

    def _save_model(self):
        self.logger.info("Saving model...")
        self.saver.save(self.sess, self.config.model_file, global_step=self.global_step)
        self.logger.info("Model save complete.")

    def _get_optimizer(self, cost, var_list=None):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.params.adam:
            # with Adam optimization: no learning rate decay
            learning_rate = tf.constant(self.params.learning_rate, dtype=tf.float32)
            sgd = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam")

        else:

            # Stochastic Gradient Descent Optimizer with exponential learning rate decay
            learning_rate = tf.train.exponential_decay(self.params.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=100000,
                                                       decay_rate=self.params.learning_decay_rate,
                                                       staircase=False,
                                                       name="learning_rate")

            sgd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")

        # this incantation ensures the BatchNorm moving mean/variance are updated with each step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = sgd.minimize(cost, var_list=var_list, name='optimizer', global_step=global_step)

        return optimizer, global_step

    def _get_job_name(self):
        # makes an identifying name for this run
        now = '{:%Y-%m-%d.%H-%M}'.format(datetime.datetime.now())
        return "%s_%s_lr_%.4f" % (self.params.model_version, now, self.params.learning_rate)
