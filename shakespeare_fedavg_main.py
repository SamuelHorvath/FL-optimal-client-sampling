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
flags.DEFINE_integer('batch_size', 1, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 100, 'Minibatch size of test data.')
flags.DEFINE_bool('importance_sampling', True, 'Importance sampling is used.')
flags.DEFINE_string('name', 'shakespeare', 'Name of the experiment.')
flags.DEFINE_integer('random_seed', 123, 'Random seed that should be used for client sampling.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

#flags.DEFINE_string('dataset_filename', 'cookup_train_1', 'Name of the cooked dataset (without suffix).')

FLAGS = flags.FLAGS

# A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='accuracy', dtype=tf.float32):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
        return super().update_state(y_true, y_pred, sample_weight)


def get_shakespeare_dataset():
    """Loads and preprocesses the Shakespeare dataset.
    Returns:
        A `(ss_train, ss_test)` tuple where `ss_train` is a
        `tff.simulation.ClientData` object representing the training data and
        `ss_test` is a single `tf.data.Dataset` representing the test data of
        all clients.
    """

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.constant(list(range(len(vocab))), dtype=tf.int64)
            ),
        default_value=0)

    def to_ids(x):
        s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids


    def split_input_target(chunk):
        input_text = tf.map_fn(lambda x: x[:-1], chunk)
        target_text = tf.map_fn(lambda x: x[1:], chunk)
        return collections.OrderedDict(
            x=input_text, y=target_text)


    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 292 for Federated Shakespeare
        return dataset.map(to_ids).shuffle(buffer_size=292).repeat(
             count=FLAGS.client_epochs_per_round).batch(
                 FLAGS.batch_size, drop_remainder=False).map(split_input_target)

    def preprocess_test_dataset(dataset):
        return dataset.map(to_ids).batch(
            FLAGS.test_batch_size, drop_remainder=False).map(split_input_target)


    ss_train, ss_test = tff.simulation.datasets.shakespeare.load_data()

    ss_train = ss_train.preprocess(preprocess_train_dataset)
    ss_test = preprocess_test_dataset(
        ss_test.create_tf_dataset_from_all_clients())
    return ss_train, ss_test


def load_model(batch_size):
    urls = {
        1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
        8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
    assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
    url = urls[batch_size]
    local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)  
    return tf.keras.models.load_model(local_file, compile=False)


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


    train_data, test_data = get_shakespeare_dataset()

    def tff_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = load_model(FLAGS.batch_size)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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

    metric_acc = FlattenedCategoricalAccuracy(name='test_accuracy')
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
