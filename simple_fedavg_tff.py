import tensorflow as tf
import tensorflow_federated as tff

from simple_fedavg_tf import build_server_broadcast_message
from simple_fedavg_tf import client_update
from simple_fedavg_tf import server_update
from simple_fedavg_tf import ServerState


def _initialize_optimizer_vars(model, optimizer):
    """Creates optimizer variables to assign the optimizer's state."""
    model_weights = model.weights
    model_delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
    # Create zero gradients to force an update that doesn't modify.
    # Force eagerly constructing the optimizer variables. Normally Keras lazily
    # creates the variables on first usage of the optimizer. Optimizers such as
    # Adam, Adagrad, or using momentum need to create a new set of variables shape
    # like the model weights.
    grads_and_vars = tf.nest.map_structure(
      lambda x, v: (tf.zeros_like(x), v), tf.nest.flatten(model_delta),
      tf.nest.flatten(model_weights.trainable))
    optimizer.apply_gradients(grads_and_vars)
    assert optimizer.variables()


class FedAvg:

    def __init__(self, model_fn, m, n, j_max, importance_sampling,
                 server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
                 client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
                 ):
        """Builds the TFF computations for optimization using federated averaging.
        Args:
        model_fn: A no-arg function that returns a
          `simple_fedavg_tf.KerasModelWrapper`.
        server_optimizer_fn: A no-arg function that returns a
          `tf.keras.optimizers.Optimizer` for server update.
        client_optimizer_fn: A no-arg function that returns a
          `tf.keras.optimizers.Optimizer` for client update.
        Returns:
        A `tff.templates.IterativeProcess`.
        """

        dummy_model = model_fn()

        @tff.tf_computation
        def server_init_tf():
            model = model_fn()
            server_optimizer = server_optimizer_fn()
            _initialize_optimizer_vars(model, server_optimizer)
            return ServerState(
                model_weights=model.weights,
                optimizer_state=server_optimizer.variables(),
                round_num=0
            )

        server_state_type = server_init_tf.type_signature.result

        model_weights_type = server_state_type.model_weights

        @tff.tf_computation(server_state_type, model_weights_type.trainable)
        def server_update_fn(server_state, model_delta):
            model = model_fn()
            server_optimizer = server_optimizer_fn()
            _initialize_optimizer_vars(model, server_optimizer)
            return server_update(model, server_optimizer, server_state, model_delta)

        @tff.tf_computation(server_state_type)
        def server_message_fn(server_state):
            return build_server_broadcast_message(server_state)

        server_message_type = server_message_fn.type_signature.result
        tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

        @tff.tf_computation(tf_dataset_type, server_message_type)
        def client_update_fn(tf_dataset, server_message):
            model = model_fn()
            client_optimizer = client_optimizer_fn()
            return client_update(model, tf_dataset, server_message, client_optimizer)

        federated_server_state_type = tff.FederatedType(server_state_type, tff.SERVER)
        federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

        @tff.tf_computation(tf.float32, tf.float32,)
        def scale(update_norm, sum_update_norms):
            if importance_sampling:
                return tf.minimum(1., tf.divide(tf.multiply(update_norm, m), sum_update_norms))
            else:
                return tf.divide(m, n)

        @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS),
                                   tff.FederatedType(tf.float32, tff.CLIENTS, True))
        def scale_on_clients(update_norm, sum_update_norms):
            return tff.federated_map(scale, (update_norm, sum_update_norms))

        @tff.tf_computation(tf.float32)
        def create_prob_message(prob):
            def f1(): return tf.stack([prob, 1.])
            def f2(): return tf.constant([0., 0.])
            prob_message = tf.cond(tf.less(prob, 1), f1, f2)
            return prob_message

        @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
        def create_prob_message_on_clients(prob):
            return tff.federated_map(create_prob_message, prob)

        @tff.tf_computation(tff.TensorType(tf.float32, (2,)))
        def compute_rescaling(prob_aggreg):
            rescaling_factor = (m - n + prob_aggreg[1])/prob_aggreg[0]
            return rescaling_factor

        @tff.federated_computation(tff.FederatedType(tff.TensorType(tf.float32, (2,)), tff.SERVER))
        def compute_rescaling_on_master(prob_aggreg):
            return tff.federated_map(compute_rescaling, prob_aggreg)

        @tff.tf_computation(tf.float32, tf.float32)
        def rescale_prob(prob, rescaling_factor):
            return tf.minimum(1., tf.multiply(prob, rescaling_factor))

        @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS),
                                   tff.FederatedType(tf.float32, tff.CLIENTS, True))
        def rescale_prob_on_clients(rob, rescaling_factor):
            return tff.federated_map(rescale_prob, (rob, rescaling_factor))

        @tff.tf_computation(tf.float32)
        def compute_weights_is_fn(prob):
            def f1(): return 1./prob
            def f2(): return 0.
            weight = tf.cond(tf.less(tf.random.uniform(()), prob), f1, f2)
            return weight

        @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
        def compute_weights_is(prob):
            return tff.federated_map(compute_weights_is_fn, prob)

        @tff.federated_computation(tff.FederatedType(model_weights_type.trainable, tff.CLIENTS),
                                   tff.FederatedType(tf.float32, tff.CLIENTS))
        def compute_round_model_delta(weights_delta, weights_denom):
            return tff.federated_mean(
                weights_delta, weight=weights_denom)

        @tff.federated_computation(federated_server_state_type,
                                   tff.FederatedType(model_weights_type.trainable, tff.SERVER))
        def update_server_state(server_state, round_model_delta):
            return tff.federated_map(server_update_fn,
                                     (server_state, round_model_delta))

        @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS),
                                   tff.FederatedType(tf.float32, tff.CLIENTS))
        def compute_loss_metric(model_output, weight_denom):
            return tff.federated_mean(
                model_output, weight=weight_denom)

        @tff.tf_computation(model_weights_type.trainable, tf.float32)
        def rescale_and_remove_fn(weights_delta, weights_is):
            return [tf.math.scalar_mul(weights_is, weights_layer_delta) for weights_layer_delta in weights_delta]

        @tff.federated_computation(tff.FederatedType(model_weights_type.trainable, tff.CLIENTS),
                                   tff.FederatedType(tf.float32, tff.CLIENTS))
        def rescale_and_remove(weights_delta, weights_is):
            return tff.federated_map(rescale_and_remove_fn,
                                     (weights_delta, weights_is))

        @tff.federated_computation(federated_server_state_type, federated_dataset_type)
        def run_gradient_computation_round(server_state, federated_dataset):
            """Orchestration logic for one round of gradient computation.
            Args:
              server_state: A `ServerState`.
              federated_dataset: A federated `tf.data.Dataset` with placement
                `tff.CLIENTS`.
            Returns:
            A tuple of updated `tf.Tensor` of clients initial probability and `ClientOutput`.
            """
            server_message = tff.federated_map(server_message_fn, server_state)
            server_message_at_client = tff.federated_broadcast(server_message)

            client_outputs = tff.federated_map(
                client_update_fn, (federated_dataset, server_message_at_client))

            update_norm_sum_weighted = tff.federated_sum(client_outputs.update_norm_weighted)
            norm_sum_clients_weighted = tff.federated_broadcast(update_norm_sum_weighted)

            prob_init = scale_on_clients(client_outputs.update_norm_weighted, norm_sum_clients_weighted)
            return prob_init, client_outputs

        @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
        def run_one_inner_loop_weights_computation(prob):
            """Orchestration logic for one round of computation.
            Args:
              prob: Probability of each client to communicate update.
            Returns:
            A tuple of updated `Probabilities` and `tf.float32` of rescaling factor.
            """

            prob_message = create_prob_message_on_clients(prob)
            prob_aggreg = tff.federated_sum(prob_message)
            rescaling_factor_master = compute_rescaling_on_master(prob_aggreg)
            rescaling_factor_clients = tff.federated_broadcast(rescaling_factor_master)
            prob = rescale_prob_on_clients(prob, rescaling_factor_clients)

            return prob, rescaling_factor_master

        @tff.federated_computation
        def server_init_tff():
            """Orchestration logic for server model initialization."""
            return tff.federated_value(server_init_tf(), tff.SERVER)

        def run_one_round(server_state, federated_dataset):
            """Orchestration logic for one round of computation.
            Args:
              server_state: A `ServerState`.
              federated_dataset: A federated `tf.data.Dataset` with placement
                `tff.CLIENTS`.
            Returns:
            A tuple of updated `ServerState` and `tf.Tensor` of average loss.
            """
            prob, client_outputs = run_gradient_computation_round(
                server_state, federated_dataset)

            if importance_sampling:
                for j in range(j_max):
                    prob, rescaling_factor = run_one_inner_loop_weights_computation(prob)
                    if rescaling_factor <= 1:
                        break

            weight_denom = [client_output.client_weight for client_output in client_outputs]
            weights_delta = [client_output.weights_delta for client_output in client_outputs]

            # rescale weights based on sampling procedure
            weights_is = compute_weights_is(prob)
            weights_delta = rescale_and_remove(weights_delta, weights_is)

            round_model_delta = compute_round_model_delta(weights_delta, weight_denom)

            server_state = update_server_state(server_state, round_model_delta)

            model_output = [client_output.model_output for client_output in client_outputs]
            round_loss_metric = compute_loss_metric(model_output, weight_denom)
            
            prob_numpy = []
            for p in prob:
                prob_numpy.append(p.numpy())

            return server_state, round_loss_metric, prob_numpy

        self.next = run_one_round
        self.initialize = server_init_tff
