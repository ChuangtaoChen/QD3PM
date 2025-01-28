import tensorflow as tf
import tensorcircuit as tc
import numpy as np


K = tc.set_backend("tensorflow")

def depolarizing_channel_no_jit_(x0, p, y, cm_state):
    x0 = tf.cast(x0, dtype=tf.complex64)
    state = tf.reshape(x0, [-1, 1])
    state_dagger = tf.linalg.adjoint(state)
    rho = tf.matmul(state, state_dagger)
    py = rho[y][y]
    return (1 - p) * cm_state + p * rho, py


def depolarizing_channel_and_sample_(x0, p, cm_state):
    x0 = tf.cast(x0, dtype=tf.complex64)
    state = tf.reshape(x0, [-1, 1])
    state_dagger = tf.linalg.adjoint(state)
    rho = tf.matmul(state, state_dagger)
    rho = (1 - p) * cm_state + p * rho
    prob = tf.cast(tf.linalg.diag_part(rho), tf.float32)
    y = tf.random.categorical(tf.math.log(tf.reshape(prob, [1, -1]) + 1e-8), num_samples=1)[0][0]
    y = tf.ensure_shape(y, [])  #

    return y, prob


def get_true_post_(y, rho, alpha, py, d):
    rho_y = tf.zeros([d, d], dtype=tf.complex64)
    index = [[y, y]]  #
    updates = [1.0 + 0.j]  #
    rho_y = tf.tensor_scatter_nd_update(rho_y, index, updates)
    rho_t_1_true = (rho_y * py * alpha + (1 - alpha) * rho / (d ** 2)) / (alpha * py + (1 - alpha) / (d ** 2))
    # rho_t_1_true = (rho_y * py * alpha + (1 - alpha) * rho ) / (alpha * py + (1 - alpha) )
    return rho_t_1_true


def PQC_(t, y, Theta, N, T, L, ansatz_connection):
    t_scaled = t / (T + 1)  #
    N_qubits = N + 1  #
    cir = tc.Circuit(N_qubits)
    index = 0

    y = tf.ensure_shape(y, [N])  #
    #
    cir.ry(0, theta=t_scaled * np.pi * 2.)

    for n in range(1, N_qubits):    # 1, N_qubits

        theta = Theta[index]
        index += 1
        theta = tf.ensure_shape(theta, [])  #
        param = theta * tf.cast(y[n - 1], tf.float32)   #
        cir.ry(n, theta=param)

    for layer in range(L):
        #
        for n in range(N_qubits):
            param = Theta[index]
            param = tf.ensure_shape(param, [])  #
            cir.rx(n, theta=param)
            index += 1
            param = Theta[index]
            param = tf.ensure_shape(param, [])  #
            cir.ry(n, theta=param)
            index += 1
            param = Theta[index]
            param = tf.ensure_shape(param, [])  #
            cir.rx(n, theta=param)
            index += 1
        # parser.add_argument('--ansatz_connection', type=str, default="all_to_all", help="all_to_all | star | chain")
        if ansatz_connection == "all_to_all":

            for i in range(N_qubits):
                for j in range(i + 1, N_qubits):
                    param = Theta[index]
                    param = tf.ensure_shape(param, [])  #
                    cir.exp1(i, j, unitary=tc.gates._zz_matrix, theta=param)
                    index += 1
        elif ansatz_connection == "chain":
            for i in range(N_qubits - 1):
                param = Theta[index]
                param = tf.ensure_shape(param, [])  #
                cir.exp1(i, i + 1, unitary=tc.gates._zz_matrix, theta=param)
                index += 1
        elif ansatz_connection == "star":
            for i in range(1, N_qubits):
                param = Theta[index]
                param = tf.ensure_shape(param, [])  #
                cir.exp1(0, i, unitary=tc.gates._zz_matrix, theta=param)
                index += 1
        else:
            assert ValueError("ansatz_connection must be either 'all_to_all' or 'chain' or 'star', get:",
                              ansatz_connection)
    state = cir.state()

    rho_pred = tc.quantum.reduced_density_matrix(state, cut=[0])
    return rho_pred


get_true_post = K.jit(K.vmap(get_true_post_, vectorized_argnums=[0, 1, 2, 3]))
PQC = K.jit(K.vmap(PQC_, vectorized_argnums=[0, 1]))
depolarizing_channel_no_jit = K.jit(K.vmap(depolarizing_channel_no_jit_, vectorized_argnums=[0, 1, 2]))
depolarizing_channel_and_sample = K.jit(K.vmap(depolarizing_channel_and_sample_, vectorized_argnums=[0, 1]))


@tf.function
def compute_input_rho(N, Theta, probs, zero_state):
    rho_encode = zero_state
    for n in range(N):
        # theta, p0, p1 = Theta[n], probs[n][0], probs[n][1]
        cos_theta = tf.cos(Theta[n] / 2)
        sin_theta = tf.sin(Theta[n] / 2)

        rho = tf.convert_to_tensor([[probs[n][0] + probs[n][1] * cos_theta * cos_theta, probs[n][1] * cos_theta * sin_theta],
               [probs[n][1] * cos_theta * sin_theta, probs[n][1] * sin_theta * sin_theta]], dtype=tf.complex64)

        rho_encode = K.kron(rho_encode, rho)
    # init_state = K.kron(zero_state, rho_encode)
    # rho_encode = rho_encode / tf.linalg.trace(rho_encode)
    return rho_encode


def sample(x_t_binary, t_list, Theta, N, T, L, ansatz_connection):

    rho = PQC(tf.cast(t_list, tf.float32), x_t_binary, Theta, N, T, L, ansatz_connection)
    prob = tf.math.real(tf.linalg.diag_part(rho))  #
    return prob


def generate_samples(num_samples, x_t_index, N, num_timesteps, Theta, L, ansatz_connection):
    binary_list = []
    binary_tensor = tf.bitwise.right_shift(tf.expand_dims(x_t_index, axis=-1), tf.range(0, N)) & 1
    binary_tensor = tf.reverse(binary_tensor, axis=[-1])
    binary_tensor = tf.ensure_shape(binary_tensor, [num_samples, N])  #

    for t in reversed(range(1, num_timesteps + 1)):
        t_list = tf.fill([num_samples], t)
        prob = sample(binary_tensor, t_list, Theta, N, num_timesteps, L, ansatz_connection)
        prob = tf.reshape(prob, [num_samples, -1])
        x_t_index = tf.random.categorical(tf.math.log(prob + 1e-8), num_samples=1, dtype=tf.int32)[:, 0]
        x_t_index = tf.ensure_shape(x_t_index, [num_samples])  #

        binary_tensor = tf.bitwise.right_shift(tf.expand_dims(x_t_index, axis=-1), tf.range(0, N)) & 1
        binary_tensor = tf.reverse(binary_tensor, axis=[-1])
        binary_tensor = tf.ensure_shape(binary_tensor, [num_samples, N])  #

        binary_list.append(tf.cast(binary_tensor, tf.int32).numpy())

    return binary_list
