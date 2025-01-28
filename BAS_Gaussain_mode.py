import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
# import matplotlib.pyplot as plt
tfd = tfp.distributions


class Discrete_Gaussian:
    def __init__(self, batch_size=16, num_dataset=100000, n_bits=10):

        self.n_bits = n_bits
        self.x_max = 2 ** self.n_bits  #
        # self.x = np.arange(1, self.x_max + 1)
        self.batch_size = batch_size

        self.nu = self.x_max / 8
        self.mu1 = (2 / 7) * self.x_max
        self.mu2 = (5 / 7) * self.x_max
        self.weights = [0.5, 0.5]
        self.probs = self.mixture_probs(self.x_max, self.mu1, self.nu, self.mu2, self.nu, self.weights)
        self.num_dataset = num_dataset
        self.dataset = self.sample_batch(self.num_dataset)

    def discrete_normal(self, loc, scale, size):

        x = tf.range(1, size + 1, dtype=tf.float32)
        probs = tf.exp(-0.5 * ((x - loc) / scale) ** 2)
        probs /= tf.reduce_sum(probs)
        return probs

    def mixture_probs(self, size, loc1, scale1, loc2, scale2, weights):
        probs1 = self.discrete_normal(loc1, scale1, size)
        probs2 = self.discrete_normal(loc2, scale2, size)
        probs = weights[0] * probs1 + weights[1] * probs2
        return probs

    def sample_batch(self, batch_size):

        probs_tensor = tf.convert_to_tensor(self.probs, dtype=tf.float32)
        categorical = tfd.Categorical(probs=probs_tensor)
        samples = categorical.sample(batch_size)
        return samples

    def samples_to_binary_array(self, samples, n_bits):

        binary_strings = [format(sample, f'0{n_bits}b') for sample in samples]

        binary_array = np.array(
            [[int(bit) for bit in binary] for binary in binary_strings],
            dtype=np.int32
        )
        return binary_array

    def get_samples(self):
        shuffled_matrix = tf.random.shuffle(self.dataset)

        batch_samples = shuffled_matrix[:self.batch_size]
        binary_samples = self.samples_to_binary_array(batch_samples.numpy(), self.n_bits)
        return binary_samples

    def get_probs(self):
        return self.probs.numpy()


def kl_divergence(p, q):

    p = np.array(p)
    q = np.array(q)


    q = np.clip(q, 1e-10, None)


    return np.sum(p * np.log(p / q))

BAS_2x2 = [

    [[1, 1],
     [0, 0]],

    [[0, 0],
     [1, 1]],


    [[1, 0],
     [1, 0]],

    [[0, 1],
     [0, 1]],

    [[1, 1],
     [1, 1]],


    [[0, 0],
     [0, 0]],
]



BAS_3x3 = [
    [[1, 1, 1],
     [0, 0, 0],
     [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 1]],

    [[1, 1, 1],
     [1, 1, 1],
     [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [1, 1, 1]],

    [[1, 1, 1],
     [0, 0, 0],
     [1, 1, 1]],

    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]],

    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],

    [[0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]],

    [[1, 1, 0],
     [1, 1, 0],
     [1, 1, 0]],

    [[0, 1, 1],
     [0, 1, 1],
     [0, 1, 1]],

    [[1, 0, 1],
     [1, 0, 1],
     [1, 0, 1]],


    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],


    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
]


if __name__ == '__main__':
    from collections import Counter
    import matplotlib.pyplot as plt
    gm = Discrete_Gaussian(32, num_dataset=50000, n_bits=8)
    print("Probabilities:")
    print(gm.get_probs())
    # for i in range(1000):

        # print(gm.get_samples())

    num_samples = 10000
    d = 2 ** 8
    batch_samples = gm.sample_batch(num_samples)
    binary_samples = gm.samples_to_binary_array(batch_samples.numpy(), 8)
    decimal_values = [int(''.join(map(str, sample.flatten())), 2) for sample in binary_samples]
    # decimal_values = [int(''.join(map(str, sample.flatten())), 2) for sample in binary_samples]


    frequency_count = Counter(decimal_values)


    all_possible_values = list(range(d))
    frequencies = [frequency_count.get(val, 0) / float(num_samples) for val in all_possible_values]
    frequencies_stand = gm.get_probs()
    print("kl=", kl_divergence(frequencies_stand, frequencies))

    plt.figure(figsize=(10, 6))


    plt.bar(all_possible_values, frequencies_stand, alpha=0.8, width=1.5, label="target", color="skyblue")
    plt.bar(all_possible_values, frequencies, alpha=0.5, width=1.5, label="trained", color="orange")


    plt.margins(x=0.05)

    plt.xlabel('3x3 Binary Pattern (as Decimal)')
    plt.ylabel('Frequency')
    plt.title('Frequency of 3x3 Binary Patterns')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("kl=", kl_divergence(frequencies_stand, frequencies))
