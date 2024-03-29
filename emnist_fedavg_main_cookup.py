import collections
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pickle
import random
import os

from tensorflow.compat.v1.keras import backend as Kbackbend

import simple_fedavg_tf
import simple_fedavg_tff

from tensorflow_federated.python.simulation import hdf5_client_data

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 151, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 5, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 32,
                     'How many clients to sample per round.')
flags.DEFINE_integer('expected_clients_per_round', 3,
                     'How many clients to communicate per round.')
flags.DEFINE_integer('j_max_iter_greedy_alg', 4,
                     'Maximum number of iteration of greedy algorithm.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 100, 'Minibatch size of test data.')
flags.DEFINE_bool('importance_sampling', True, 'Importance sampling is used.')
flags.DEFINE_string('name', 'emnist', 'Name of the experiment.')
flags.DEFINE_integer('random_seed', 123, 'Random seed that should be used for client sampling.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

flags.DEFINE_string('dataset_filename', 'cookup_train_1', 'Name of the cooked dataset (without suffix).')

FLAGS = flags.FLAGS


def get_emnist_dataset(dataset_filename):
    """Loads and preprocesses the EMNIST dataset.
    Returns:
        A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
        `tff.simulation.ClientData` object representing the training data and
        `emnist_test` is a single `tf.data.Dataset` representing the test data of
        all clients.
    """
    emnist_train = hdf5_client_data.HDF5ClientData(f'./dataset/{dataset_filename}.h5')
    emnist_test = hdf5_client_data.HDF5ClientData('./dataset/test.h5')

    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label'])

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
             count=FLAGS.client_epochs_per_round).batch(
                 FLAGS.batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(
            FLAGS.test_batch_size, drop_remainder=False)

    emnist_train = emnist_train.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        emnist_test.create_tf_dataset_from_all_clients())
    return emnist_train, emnist_test


def create_original_fedavg_cnn_model(only_digits=True):
    """The CNN model used in https://arxiv.org/abs/1602.05629.
    This function is duplicated from research/optimization/emnist/models.py to
    make this example completely stand-alone.
    Args:
        only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only EMNIST dataset. If False, uses 62 outputs for the larger
        dataset.
    Returns:
        An uncompiled `tf.keras.Model`.
    """
    data_format = 'channels_last'
    input_shape = [28, 28, 1]

    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=FLAGS.random_seed))

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=FLAGS.random_seed)),
        tf.keras.layers.Dense(10 if only_digits else 62,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=FLAGS.random_seed)),
        tf.keras.layers.Activation(tf.nn.softmax),
    ])

    return model


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    os.environ['PYTHONHASHSEED'] = str(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    Kbackbend.set_session(sess)

    train_data, test_data = get_emnist_dataset(FLAGS.dataset_filename)

    def tff_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = create_original_fedavg_cnn_model(only_digits=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        return simple_fedavg_tf.KerasModelWrapper(keras_model,
                                                  test_data.element_spec, loss)

    # iterative_process = simple_fedavg_tff.build_federated_averaging_process(
    #     tff_model_fn, FLAGS.expected_clients_per_round, FLAGS.train_clients_per_round,
    #     FLAGS.j_max_iter_greedy_alg, FLAGS.importance_sampling, server_optimizer_fn, client_optimizer_fn)
    # server_state = iterative_process.initialize()

    federated_averaging = simple_fedavg_tff.FedAvg(
            tff_model_fn, FLAGS.expected_clients_per_round, FLAGS.train_clients_per_round,
            FLAGS.j_max_iter_greedy_alg, FLAGS.importance_sampling, server_optimizer_fn, client_optimizer_fn)
    server_state = federated_averaging.initialize()

    metric_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    model = tff_model_fn()

    train_loss = []
    val_acc = []
    probs = []

    clients = train_data.client_ids

    np.random.seed(FLAGS.random_seed)
    sampled_clients_list = [np.random.choice(
        len(clients),
        size=FLAGS.train_clients_per_round,
        replace=False) for _ in range(FLAGS.total_rounds)]

    for round_num in range(FLAGS.total_rounds):
        sampled_clients = [clients[i] for i in sampled_clients_list[round_num]]
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]
        # server_state, train_metrics = iterative_process.next(
        #     server_state, sampled_train_data)

        server_state, train_metrics, prob = federated_averaging.next(
            server_state, sampled_train_data)

        print(f'Round {round_num} training loss: {train_metrics}')
        train_loss.append(train_metrics)
        probs.append(prob)

        if round_num % FLAGS.rounds_per_eval == 0:
            model.from_weights(server_state.model_weights)
            accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, test_data,
                                                       metric_acc)
            print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')
            val_acc.append(accuracy)

    with open(f'./tff_save/{FLAGS.name}_valacc.pk', 'wb') as f:
        pickle.dump(val_acc, f)
    with open(f'./tff_save/{FLAGS.name}_trainloss.pk', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(f'./tff_save/{FLAGS.name}_probs.pk', 'wb') as f:
        pickle.dump(probs, f)


if __name__ == '__main__':
    app.run(main)
