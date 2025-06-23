import tensorflow as tf
import tensorcircuit as tc
import numpy as np


K = tc.set_backend("tensorflow")

def depolarizing_channel_no_jit_(x0, p, y, cm_state):
    x0 = tf.cast(x0, dtype=tf.complex64)
    state = tf.reshape(x0, [-1, 1])
    state_dagger = tf.linalg.adjoint(state)
    rho = tf.matmul(state, state_dagger)
    rho_t_1 = (1 - p) * cm_state + p * rho
    py = tf.gather_nd(rho_t_1, [[y, y]])
    return rho_t_1, py


#  实现一步跳过去，利用alpha_bar.得到t时刻的态并采样
def depolarizing_channel_and_sample_(x0, p, cm_state):
    x0 = tf.cast(x0, dtype=tf.complex64)
    state = tf.reshape(x0, [-1, 1])
    state_dagger = tf.linalg.adjoint(state)
    rho = tf.matmul(state, state_dagger)
    rho = (1 - p) * cm_state + p * rho
    prob = tf.cast(tf.linalg.diag_part(rho), tf.float32)
    y = tf.random.categorical(tf.math.log(tf.reshape(prob, [1, -1]) + 1e-8), num_samples=1)[0][0]
    y = tf.ensure_shape(y, [])  # 确保 y 是标量

    return y, prob


def get_true_post_(y, rho, alpha, py, d):
    rho_y = tf.zeros([d, d], dtype=tf.complex64)
    index = [[y, y]]  # 第 y 行第 y 列
    updates = [1.0 + 0.j]  # 更新值为 1
    rho_y = tf.tensor_scatter_nd_update(rho_y, index, updates)
    rho_t_1_true = (rho_y * py * alpha + (1 - alpha) * rho / (d ** 2)) / (alpha * py + (1 - alpha) / (d ** 2))
    # rho_t_1_true = (rho_y * py * alpha + (1 - alpha) * rho ) / (alpha * py + (1 - alpha) )
    return rho_t_1_true


def PQC_(t, y, Theta, N, T, L, ansatz_connection):
    t_scaled = t / (T + 1)  # 将 t 缩放到 [0, 1]
    N_qubits = N + 1  # 增加一个辅助量子比特  #
    cir = tc.Circuit(N_qubits)
    index = 0

    y = tf.ensure_shape(y, [N])  # 确保 y 的形状是 [N]
    # 量子线路的主体部分：增加深度和纠缠
    cir.ry(0, theta=t_scaled * np.pi * 2.)

    for n in range(1, N_qubits):    # 1, N_qubits

        theta = Theta[index]
        index += 1
        theta = tf.ensure_shape(theta, [])  # 确保 theta 是标量
        param = theta * tf.cast(y[n - 1], tf.float32)   #
        cir.ry(n, theta=param)

    for layer in range(L):
        # 参数化旋转门
        for n in range(N_qubits):
            param = Theta[index]
            param = tf.ensure_shape(param, [])  # 确保 theta 是标量
            cir.rx(n, theta=param)
            index += 1
            param = Theta[index]
            param = tf.ensure_shape(param, [])  # 确保 theta 是标量
            cir.ry(n, theta=param)
            index += 1
            param = Theta[index]
            param = tf.ensure_shape(param, [])  # 确保 theta 是标量
            cir.rx(n, theta=param)
            index += 1
        # parser.add_argument('--ansatz_connection', type=str, default="all_to_all", help="all_to_all | star | chain")
        if ansatz_connection == "all_to_all":
            # 全连接纠缠层
            for i in range(N_qubits):
                for j in range(i + 1, N_qubits):
                    param = Theta[index]
                    param = tf.ensure_shape(param, [])  # 确保 theta 是标量
                    cir.exp1(i, j, unitary=tc.gates._zz_matrix, theta=param)
                    index += 1
        elif ansatz_connection == "chain":
            for i in range(N_qubits - 1):
                param = Theta[index]
                param = tf.ensure_shape(param, [])  # 确保 theta 是标量
                cir.exp1(i, i + 1, unitary=tc.gates._zz_matrix, theta=param)
                index += 1
        elif ansatz_connection == "star":
            for i in range(1, N_qubits):
                param = Theta[index]
                param = tf.ensure_shape(param, [])  # 确保 theta 是标量
                cir.exp1(0, i, unitary=tc.gates._zz_matrix, theta=param)
                index += 1
        else:
            assert ValueError("ansatz_connection must be either 'all_to_all' or 'chain' or 'star', get:",
                              ansatz_connection)
    state = cir.state()
    # 对预测量子比特部分裁剪，得到预测的密度矩阵
    rho_pred = tc.quantum.reduced_density_matrix(state, cut=[0])
    return rho_pred


get_true_post = K.jit(K.vmap(get_true_post_, vectorized_argnums=[0, 1, 2, 3]))
PQC = K.jit(K.vmap(PQC_, vectorized_argnums=[0, 1]))
depolarizing_channel_no_jit = K.jit(K.vmap(depolarizing_channel_no_jit_, vectorized_argnums=[0, 1, 2]))
depolarizing_channel_and_sample = K.jit(K.vmap(depolarizing_channel_and_sample_, vectorized_argnums=[0, 1]))


def sample(x_t_binary, t_list, Theta, N, T, L, ansatz_connection):

    rho = PQC(tf.cast(t_list, tf.float32), x_t_binary, Theta, N, T, L, ansatz_connection)
    prob = tf.math.real(tf.linalg.diag_part(rho))  # 提取测量概率
    return prob


def generate_samples(num_samples, x_t_index, N, num_timesteps, Theta, L, ansatz_connection):
    binary_list = []
    binary_tensor = tf.bitwise.right_shift(tf.expand_dims(x_t_index, axis=-1), tf.range(0, N)) & 1
    binary_tensor = tf.reverse(binary_tensor, axis=[-1])
    binary_tensor = tf.ensure_shape(binary_tensor, [num_samples, N])  # 确保形状明确

    for t in reversed(range(1, num_timesteps + 1)):
        t_list = tf.fill([num_samples], t)
        prob = sample(binary_tensor, t_list, Theta, N, num_timesteps, L, ansatz_connection)
        prob = tf.reshape(prob, [num_samples, -1])
        x_t_index = tf.random.categorical(tf.math.log(prob + 1e-8), num_samples=1, dtype=tf.int32)[:, 0]
        x_t_index = tf.ensure_shape(x_t_index, [num_samples])  # 确保形状明确

        binary_tensor = tf.bitwise.right_shift(tf.expand_dims(x_t_index, axis=-1), tf.range(0, N)) & 1
        binary_tensor = tf.reverse(binary_tensor, axis=[-1])
        binary_tensor = tf.ensure_shape(binary_tensor, [num_samples, N])  # 再次确保形状明确

        binary_list.append(tf.cast(binary_tensor, tf.int32).numpy())

    return binary_list
