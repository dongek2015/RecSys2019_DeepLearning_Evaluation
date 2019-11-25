#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
from argparse import ArgumentParser

import tensorflow as tf
import pandas as pd
import numpy as np

from neumf import ncf_model_ops
from input_pipeline import DataGenerator

from logger.logger import LOGGER
from logger.autologging import log_args

def parse_args():
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description="Train a Neural Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train for')
    parser.add_argument('-b', '--batch-size', type=int, default=1048576,
                        help='number of examples for each iteration')
    parser.add_argument('--valid-users-per-batch', type=int, default=5000,
                        help='Number of users tested in each evaluation batch')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0045,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='manually set random seed for random number generation')
    parser.add_argument('--target', '-t', type=float, default=0.9562,
                        help='stop training early at target')
    parser.add_argument('--fp16', action='store_true', dest='amp',
                        help='enable half-precision computations using automatic mixed precision \
                              (only available in supported containers)')
    parser.add_argument('--manual-fp16', action='store_true', dest='fp16',
                        help='manually enable mixed precision using code changes')
    parser.add_argument('--xla', action='store_true',
                        help='enable TensorFlow XLA (Accelerated Linear Algebra)')
    parser.add_argument('--valid-negative', type=int, default=100,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--beta1', '-b1', type=float, default=0.25,
                        help='beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.5,
                        help='beta2 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='epsilon for Adam')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--loss-scale', default=8192, type=int,
                        help='Loss scale value to use when manually enabling mixed precision')
    parser.add_argument('--checkpoint-dir', default='/data/checkpoints/', type=str,
                        help='Path to the store the result checkpoint file for training, \
                              or to read from for evaluation')
    parser.add_argument('--load-checkpoint-path', default=None, type=str,
                        help='Path to the checkpoint for initialization. If None will initialize with random weights')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation, \
                              otherwise full training will be performed')
    parser.add_argument('--eval-after', type=int, default=8,
                        help='Perform evaluations only after this many epochs')
    parser.add_argument('--verbose', action='store_true',
                        help='Log the performance and accuracy after every epoch')

    return parser.parse_args()


 def main():
    """
    Run training/evaluation
    """
    script_start = time.time()
    args = parse_args()

    log_args(args)

    if args.seed is not None:
        tf.random.set_random_seed(args.seed)
        np.random.seed(args.seed)

    if args.amp:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    if "TF_ENABLE_AUTO_MIXED_PRECISION" in os.environ \
       and os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] == "1":
        args.fp16 = False

    # directory to store/read final checkpoint
    if args.mode == 'train':
        print("Saving best checkpoint to {}".format(args.checkpoint_dir))
    else:
        print("Reading checkpoint: {}".format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'model.ckpt')

    # Load converted data and get statistics
    train_df = pd.read_csv(args.data+'/math_train.csv')
    test_df = pd.read_csv(args.data+'/math_val.csv')
    nb_users, nb_items = (train_df.max() + 1)[:2]

    # Extract train and test feature tensors from dataframe
    train_users = train_df.iloc[:, 0].values.astype(np.int32)
    train_items = train_df.iloc[:, 1].values.astype(np.int32)
    test_users = test_df.iloc[:, 0].values.astype(np.int32)
    test_items = test_df.iloc[:, 1].values.astype(np.int32)

    # negative become 1, as it is more rare
    train_labels = (train_data.answer.values != 1).astype(np.float32)
    test_labels = (test_data.answer.values != 1).astype(np.float32)

    # Create and run Data Generator in a separate thread
    data_generator = DataGenerator(
        args.seed,
        nb_users,
        nb_items,
        train_users,
        train_items,
        train_labels,
        args.batch_size,
        test_users,
        test_items,
        test_labels,
        args.valid_users_per_batch,
        )

    # Create tensorflow session and saver
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)

    # Input tensors
    users = tf.placeholder(tf.int32, shape=(None,))
    items = tf.placeholder(tf.int32, shape=(None,))
    labels = tf.placeholder(tf.int32, shape=(None,))
    is_dup = tf.placeholder(tf.float32, shape=(None,))
    dropout = tf.placeholder_with_default(args.dropout, shape=())
    # Model ops and saver
    hit_rate, ndcg, eval_op, train_op = ncf_model_ops(
        users,
        items,
        labels,
        is_dup,
        params={
            'fp16': args.fp16,
            'val_batch_size': args.valid_negative+1,
            'top_k': args.topk,
            'learning_rate': args.learning_rate,
            'beta_1': args.beta1,
            'beta_2': args.beta2,
            'epsilon': args.eps,
            'num_users': nb_users,
            'num_items': nb_items,
            'num_factors': args.factors,
            'mf_reg': 0,
            'layer_sizes': args.layers,
            'layer_regs': [0. for i in args.layers],
            'dropout': dropout,
            'sigmoid': True,
            'loss_scale': args.loss_scale
        },
        mode='TRAIN' if args.mode == 'train' else 'EVAL'
    )
    saver = tf.train.Saver()

    # Accuracy metric tensors
    hr_sum = tf.get_default_graph().get_tensor_by_name('neumf/hit_rate/total:0')
    hr_cnt = tf.get_default_graph().get_tensor_by_name('neumf/hit_rate/count:0')
    ndcg_sum = tf.get_default_graph().get_tensor_by_name('neumf/ndcg/total:0')
    ndcg_cnt = tf.get_default_graph().get_tensor_by_name('neumf/ndcg/count:0')

    # Prepare evaluation data
    data_generator.prepare_eval_data()

    if args.load_checkpoint_path:
        saver.restore(sess, args.load_checkpoint_path)
    else:
        # Manual initialize weights
        sess.run(tf.global_variables_initializer())

    # If test mode, run one eval
    if args.mode == 'test':
        sess.run(tf.local_variables_initializer())
        eval_start = time.time()
        for user_batch, item_batch, dup_batch \
            in zip(data_generator.eval_users, data_generator.eval_items, data_generator.dup_mask):
            sess.run(
                eval_op,
                feed_dict={
                    users: user_batch,
                    items: item_batch,
                    is_dup:dup_batch, dropout: 0.0
                }
            )
        eval_duration = time.time() - eval_start

        # Report results
        hit_rate_sum = sess.run(hr_sum)
        hit_rate_cnt = sess.run(hr_cnt)
        ndcg_sum = sess.run(ndcg_sum)
        ndcg_cnt = sess.run(ndcg_cnt)

        hit_rate = hit_rate_sum / hit_rate_cnt
        ndcg = ndcg_sum / ndcg_cnt

        LOGGER.log("Eval Time: {:.4f}, HR: {:.4f}, NDCG: {:.4f}"
                   .format(eval_duration, hit_rate, ndcg))

        eval_throughput = pos_test_users.shape[0] * (args.valid_negative + 1) / eval_duration
        LOGGER.log('Average Eval Throughput: {:.4f}'.format(eval_throughput))
        return

    # Performance Metrics
    train_times = list()
    eval_times = list()
    # Accuracy Metrics
    first_to_target = None
    time_to_train = 0.0
    best_hr = 0
    best_epoch = 0
    # Buffers for local metrics
    local_hr_sum = np.ones(1)
    local_hr_count = np.ones(1)
    local_ndcg_sum = np.ones(1)
    local_ndcg_count = np.ones(1)

    # Begin training
    begin_train = time.time()
    LOGGER.log("Begin Training. Setup Time: {}".format(begin_train - script_start))
    for epoch in range(args.epochs):
        # Train for one epoch
        train_start = time.time()
        data_generator.prepare_train_data()
        for user_batch, item_batch, label_batch \
            in zip(data_generator.train_users_batches,
                   data_generator.train_items_batches,
                   data_generator.train_labels_batches):
            sess.run(
                train_op,
                feed_dict={
                    users: user_batch.get(),
                    items: item_batch.get(),
                    labels: label_batch.get()
                }
            )
        train_duration = time.time() - train_start
        ## Only log "warm" epochs
        if epoch >= 1:
            train_times.append(train_duration)
        # Evaluate
        if epoch > args.eval_after:
            eval_start = time.time()
            sess.run(tf.local_variables_initializer())
            for user_batch, item_batch, dup_batch \
                in zip(data_generator.eval_users,
                       data_generator.eval_items,
                       data_generator.dup_mask):
                sess.run(
                    eval_op,
                    feed_dict={
                        users: user_batch,
                        items: item_batch,
                        is_dup: dup_batch,
                        dropout: 0.0
                    }
                )
            # Compute local metrics
            local_hr_sum[0] = sess.run(hr_sum)
            local_hr_count[0] = sess.run(hr_cnt)
            local_ndcg_sum[0] = sess.run(ndcg_sum)
            local_ndcg_count[0] = sess.run(ndcg_cnt)
            # Calculate metrics
            hit_rate = local_hr_sum[0] / local_hr_count[0]
            ndcg = local_ndcg_sum[0] / local_ndcg_count[0]

            eval_duration = time.time() - eval_start
            ## Only log "warm" epochs
            if epoch >= 1:
                eval_times.append(eval_duration)

            if args.verbose:
                log_string = "Epoch: {:02d}, Train Time: {:.4f}, Eval Time: {:.4f}, HR: {:.4f}, NDCG: {:.4f}"
                LOGGER.log(
                    log_string.format(
                        epoch,
                        train_duration,
                        eval_duration,
                        hit_rate,
                        ndcg
                    )
                )

            # Update summary metrics
            if hit_rate > args.target and first_to_target is None:
                first_to_target = epoch
                time_to_train = time.time() - begin_train
            if hit_rate > best_hr:
                best_hr = hit_rate
                best_epoch = epoch
                time_to_best =  time.time() - begin_train
                if not args.verbose:
                    log_string = "New Best Epoch: {:02d}, Train Time: {:.4f}, Eval Time: {:.4f}, HR: {:.4f}, NDCG: {:.4f}"
                    LOGGER.log(
                        log_string.format(
                            epoch,
                            train_duration,
                            eval_duration,
                            hit_rate,
                            ndcg
                        )
                    )
                # Save, if meets target
                if hit_rate > args.target:
                    saver.save(sess, final_checkpoint_path)

    # Final Summary
    train_times = np.array(train_times)
    train_throughputs = pos_train_users.shape[0]*(args.negative_samples+1) / train_times
    eval_times = np.array(eval_times)
    eval_throughputs = pos_test_users.shape[0]*(args.valid_negative+1) / eval_times
    LOGGER.log(' ')

    LOGGER.log('batch_size: {}'.format(args.batch_size))
    LOGGER.log('AMP: {}'.format(1 if args.amp else 0))
    LOGGER.log('seed: {}'.format(args.seed))
    LOGGER.log('Minimum Train Time per Epoch: {:.4f}'.format(np.min(train_times)))
    LOGGER.log('Average Train Time per Epoch: {:.4f}'.format(np.mean(train_times)))
    LOGGER.log('Average Train Throughput:     {:.4f}'.format(np.mean(train_throughputs)))
    LOGGER.log('Minimum Eval Time per Epoch:  {:.4f}'.format(np.min(eval_times)))
    LOGGER.log('Average Eval Time per Epoch:  {:.4f}'.format(np.mean(eval_times)))
    LOGGER.log('Average Eval Throughput:      {:.4f}'.format(np.mean(eval_throughputs)))
    LOGGER.log('First Epoch to hit:           {}'.format(first_to_target))
    LOGGER.log('Time to Train:                {:.4f}'.format(time_to_train))
    LOGGER.log('Time to Best:                 {:.4f}'.format(time_to_best))
    LOGGER.log('Best HR:                      {:.4f}'.format(best_hr))
    LOGGER.log('Best Epoch:                   {}'.format(best_epoch))

    sess.close()
    return

if __name__ == '__main__':
    main()
