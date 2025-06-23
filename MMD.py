import tensorflow as tf
import numpy as np


class RBFMMD2TF(tf.Module):
    def __init__(self, sigma, num_bit, is_binary):
        super(RBFMMD2TF, self).__init__()
        self.sigma = sigma
        self.num_bit = num_bit
        self.is_binary = is_binary
        # 生成所有可能的 bitstrings，表示为整数
        self.basis = tf.constant(np.arange(2**num_bit, dtype='int32'), dtype=tf.int32)
        # 预计算核矩阵
        self.K = self.rbf_kernel(self.basis, self.basis, self.sigma, self.is_binary)

    def rbf_kernel(self, x, y, sigma, is_binary):
        if is_binary:
            # 获取分布的大小
            d = tf.shape(x)[0]
            dy = tf.shape(y)[0]
            # 初始化 Hamming 距离矩阵为 0
            dx2 = tf.zeros((d, dy), dtype=tf.int32)
            for i in range(self.num_bit):
                # 提取 x 和 y 的第 i 位
                bit_x = tf.bitwise.bitwise_and(tf.bitwise.right_shift(x[:, None], i), 1)  # [d, 1]
                bit_y = tf.bitwise.bitwise_and(tf.bitwise.right_shift(y[None, :], i), 1)  # [1, dy]
                # 计算不同的位
                diff = tf.not_equal(bit_x, bit_y)  # [d, dy], bool
                # 累加 Hamming 距离
                dx2 += tf.cast(diff, tf.int32)  # [d, dy]
        else:
            # 对于非二进制数据，计算欧氏距离平方
            x_f = tf.cast(tf.expand_dims(x, axis=1), tf.float32)  # [d, 1]
            y_f = tf.cast(tf.expand_dims(y, axis=0), tf.float32)  # [1, dy]
            dx = x_f - y_f  # [d, dy]
            dx2 = tf.reduce_sum(tf.square(dx), axis=-1)  # [d, dy]
        # 计算 RBF 核
        return self._rbf_kernel(dx2, sigma)  # [d, dy]

    def _rbf_kernel(self, dx2, sigma):

        if isinstance(sigma, list):
            K = tf.zeros_like(dx2, dtype=tf.float32)
            for g in sigma:
                g = 1.0 / (2.0 * g)
                K += tf.exp(-g * tf.cast(dx2, tf.float32))
            K /= len(sigma)
        else:
            gamma = 1.0 / (2.0 * sigma)
            K = tf.exp(-gamma * tf.cast(dx2, tf.float32))
        return K  # [d, dy]

    @tf.function
    def __call__(self, px, py):
        '''
        Args:
            px (2D tensor): probability for distribution P, shape [batch, d]
            py (2D tensor): probability for distribution Q, shape [batch, d]

        Returns:
            1D tensor: MMD loss for each batch, shape [batch]
        '''
        # 计算 pxy = px - py，形状 [batch, d]
        pxy = px - py  # [batch, d]
        return self.kernel_expect(pxy)  # [batch]

    def kernel_expect(self, pxy):
        '''
        Args:
            pxy (2D tensor): p(x) - q(y), shape [batch, d]

        Returns:
            1D tensor: MMD loss for each batch, shape [batch]
        '''
        # 使用 tf.einsum 计算每个批次的 MMD
        # 'bi,ij,bj->b' 表示对每个批次 i，计算 pxy[i] @ K @ pxy[i]^T
        res = tf.einsum('bi,ij,bj->b', pxy, self.K, pxy)  # [batch]
        return res


# 使用示例
if __name__ == "__main__":
    num_bit = 10  # 位数，d=2^num_bit

    is_binary = True

    mmd = RBFMMD2TF([1.0, 2.0], num_bit, is_binary)

    # 定义两个概率分布 P 和 Q
    d = 2 ** num_bit
    # px = tf.constant([2, 1.0 / d] * d, dtype=tf.float32)  # 均匀分布
    px = tf.random.normal([4, d])
    py = tf.random.normal([4, d], mean=1.0)
    px = tf.nn.softmax(px, -1)
    py = tf.nn.softmax(py, -1)

    # 计算 MMD
    mmd_loss = mmd(px, py)
    print("MMD Loss:", mmd_loss.numpy())
