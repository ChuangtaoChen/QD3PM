import numpy as np
import tensorflow as tf
from quantum import K, get_true_post, PQC, depolarizing_channel_no_jit, depolarizing_channel_and_sample


def compute_g(t, T, s):
    return np.cos(((t / T + s) / (1 + s)) * (np.pi / 2)) ** 2

def compute_alpha_bar(T, s):
    g_0 = compute_g(0, T, s)
    alpha_bar = np.array([compute_g(t, T, s) / g_0 for t in range(T + 1)])
    return alpha_bar

def compute_alpha_t(alpha_bar):
    alpha_t = np.array([alpha_bar[t] / alpha_bar[t - 1] for t in range(1, len(alpha_bar))])
    return alpha_t


def binary_to_decimal(binary_tensor, N):
    """
    """
    powers_of_two = tf.constant([2 ** i for i in reversed(range(N))], dtype=tf.int32)
    decimal_tensor = tf.reduce_sum(binary_tensor * powers_of_two, axis=1)
    return decimal_tensor


def decimal_to_one_hot(decimal_tensor, num_classes):

    one_hot_tensor = tf.one_hot(decimal_tensor, depth=num_classes, dtype=tf.float32)
    return one_hot_tensor


@tf.function
def train_step(x0, t, N, d, alpha_bar, alpha_t, Theta, opt, cm_state, T, batch_size, mmd_loss, L, ansatz_connection):

    x0 = tf.reshape(x0, [-1, N])  # [batch_size, N]
    decimal_data = binary_to_decimal(x0, N)  # [batch_size]
    x0_one_hot = decimal_to_one_hot(decimal_data, num_classes=d)  # [batch_size, d]

    #
    alpha_bar_t_list = tf.gather(alpha_bar, t)
    alpha_bar_t_1_list = tf.gather(alpha_bar, t - 1)


    y, _ = depolarizing_channel_and_sample(x0_one_hot, alpha_bar_t_list, cm_state)
    y = tf.cast(y, tf.int32)
    y_binary = tf.reverse(tf.bitwise.right_shift(tf.expand_dims(y, axis=-1), tf.range(0, N)) & 1, axis=[-1])

    rho_t_1, py = depolarizing_channel_no_jit(x0_one_hot, alpha_bar_t_1_list, y, cm_state)

    alpha_t_list = tf.gather(alpha_t, t - 1)
    updated_rho = get_true_post(y, rho_t_1, alpha_t_list, py, d)
    prob_true_t = tf.math.real(tf.linalg.diag_part(updated_rho))

    t1 = tf.ones([batch_size], tf.int32)
    alpha_bar_t1_list = tf.gather(alpha_bar, t1)

    y1, _ = depolarizing_channel_and_sample(x0_one_hot, alpha_bar_t1_list, cm_state)
    y1 = tf.cast(y1, tf.int32)
    y1_binary = tf.reverse(tf.bitwise.right_shift(tf.expand_dims(y1, axis=-1), tf.range(0, N)) & 1, axis=[-1])

    rho_t_0, py1 = depolarizing_channel_no_jit(x0_one_hot, tf.gather(alpha_bar, t1 - 1), y1, cm_state) #

    prob_true_0 = tf.math.real(tf.linalg.diag_part(rho_t_0))
    with tf.GradientTape() as tape:
        tape.watch(Theta)
        # t, y, Theta, N, T, L, ansatz_connection
        rho_pred_t = PQC(tf.cast(t, tf.float32), y_binary, Theta, N, T, L, ansatz_connection)
        prob_pred_t = tf.math.real(tf.linalg.diag_part(rho_pred_t))
        lt = tf.reduce_mean(mmd_loss(prob_pred_t, prob_true_t))

        rho_pred_0 = PQC(tf.cast(t1, tf.float32), y1_binary, Theta, N, T, L, ansatz_connection)
        prob_pred_0 = tf.math.real(tf.linalg.diag_part(rho_pred_0))
        l0 = tf.reduce_mean(mmd_loss(prob_pred_0, prob_true_0))

        loss = l0 + lt

    gradients = tape.gradient(loss, Theta)
    opt.apply_gradients([(grad, var) for grad, var in zip([gradients], [Theta])])
    return loss, lt, l0


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=0.01, beta=0.005, min_lr=1e-5):

        self.initial_lr = initial_lr
        self.beta = beta
        self.min_lr = min_lr

    def __call__(self, step):

        step = tf.cast(step, tf.float32)

        decayed_lr = self.initial_lr * tf.math.exp(-self.beta * step)

        return tf.math.maximum(decayed_lr, self.min_lr)

