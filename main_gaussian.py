import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import argparse

from collections import Counter
import pickle
from BAS_Gaussain_mode import Discrete_Gaussian
import time
from MMD import RBFMMD2TF
from quantum import *
from utils import *

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("Use GPU")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"error: {e}")
else:
    print("Use CPU")


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--N', type=int, default=8)
parser.add_argument('--num_dataset', type=int, default=50000)

parser.add_argument('--num_timesteps', type=int, default=30)

parser.add_argument('--sigma', type=int, default=10000, help="10000: hybrid")

parser.add_argument('--ansatz_connection', type=str, default="all_to_all", help="all_to_all | star | chain")
parser.add_argument('--layer', type=int, default=6)

parser.add_argument('--epochs', type=int, default=6000)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--lr_decay_type', type=str, default="cos")
parser.add_argument('--initial_lr', type=float, default=0.001)  #
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--min_lr', type=float, default=1e-5)

parser.add_argument('--decay_steps', type=int, default=5000)

parser.add_argument('--num_samples', type=int, default=100000)

parser.add_argument('--fast_flat', type=bool, default=False)
parser.add_argument('--split_flat', type=bool, default=False)
parser.add_argument('--check_point', type=bool, default=True)
args = parser.parse_args()



np.random.seed(args.seed)
tf.random.set_seed(args.seed)


if args.sigma == 10000:
    sigma = [0.01 * args.N, 0.1 * args.N, 0.25 * args.N, 0.5 * args.N, 1.0 * args.N, 10 * args.N]
else:
    sigma = args.sigma

mmd_loss = RBFMMD2TF(sigma=sigma, num_bit=args.N, is_binary=True)


class Model:
    def __init__(self, args):
        self.args = args
        self.num_classes = 2
        self.num_timesteps = args.num_timesteps
        self.L, self.ansatz_connection = args.layer, args.ansatz_connection
        self.batch_size = args.batch_size
        self.N = args.N
        self.d = 2 ** self.N

        cor = (self.N + 1) * self.N / 2 if args.ansatz_connection == "all_to_all" else self.N + 1
        num_theta = args.layer * (3 * (self.N + 1) + cor) + self.N + 1
        std = np.sqrt(1 / (4 * ((args.layer - 1) * 3 + 2)))
        self.Theta = tf.Variable(tf.random.normal(shape=[int(num_theta)], stddev=std, dtype=tf.float32))

        self.alpha_bar = compute_alpha_bar(args.num_timesteps, s=0.008)
        self.alpha_t = compute_alpha_t(self.alpha_bar)

        self.alpha_bar = tf.convert_to_tensor(self.alpha_bar, dtype=tf.complex64)
        self.alpha_t = tf.convert_to_tensor(self.alpha_t, dtype=tf.complex64)

        self.cm_state = tf.convert_to_tensor(np.identity(2 ** self.N, dtype=np.complex64) / 2 ** self.N)
        if args.lr_decay_type == "exp":
            schedules = CustomLearningRateSchedule(initial_lr=args.initial_lr, beta=args.beta, min_lr=args.min_lr)
            self.opt = tf.keras.optimizers.Adam(learning_rate=schedules)
        elif args.lr_decay_type == "cos":
            schedules = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.initial_lr,
                                                                  decay_steps=args.decay_steps,
                                                                  alpha=args.min_lr / args.initial_lr)
            self.opt = tf.keras.optimizers.Adam(learning_rate=schedules)
        else:
            self.opt = tf.keras.optimizers.Adam(learning_rate=args.initial_lr)

        self.gaussian_model = Discrete_Gaussian(batch_size=args.batch_size, n_bits=self.N, num_dataset=args.num_dataset)

        self.frequencies_stand = self.gaussian_model.get_probs()
        self.all_possible_values = list(range(self.d))
        if args.check_point:
            self.check_point = [0, 1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000,
                                1568, 2446, 3832, 5999]
        else:
            self.check_point = []
        self.cut_index = [[i for i in range(args.N) if i != n] for n in range(args.N)]

    def forward(self):
        mean_loss = []

        result_path = None
        epoch_list = []

        lt_list, l0_list = [], []
        kl_list = []
        kl_cp_list = []
        for epoch in range(self.args.epochs):
            t = tf.random.uniform(shape=[self.batch_size], minval=2, maxval=self.num_timesteps + 1, dtype=tf.int32)

            x_batch = self.gaussian_model.get_samples()     # (self.batch_size, 10)

            loss, lt, l0 = train_step(x_batch, t, self.N, self.d, self.alpha_bar,
                                      self.alpha_t, self.Theta, self.opt, self.cm_state, self.num_timesteps,
                                      self.batch_size, mmd_loss, self.L, self.ansatz_connection)
            lt_list.append(lt.numpy())
            l0_list.append(l0.numpy())
            mean_loss.append(loss.numpy())

            if (epoch + 1) % 500 == 0 or epoch == args.epochs - 1 or epoch == 0 or epoch in self.check_point:
                print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}, lt:{lt.numpy():.4f}, l0:{l0.numpy():.4f}")

                result_path, kl = self.sample_and_evaluate(num_samples=args.num_samples, epoch=epoch, result_path=result_path)

                if (epoch + 1) % 500 == 0 or epoch == args.epochs - 1 or epoch == 0:
                    epoch_list.append(epoch)
                    kl_list.append(kl)
                if epoch in self.check_point:
                    kl_cp_list.append(kl)

        np.savetxt(result_path + "/epoch_list.txt", epoch_list, fmt='%d')
        np.savetxt(result_path + "/kl_list.txt", kl_list, fmt='%.6f')
        np.savetxt(result_path + "/kl_check_point_list.txt", kl_cp_list, fmt='%.6f')

        np.savetxt(result_path + "/lt_list.txt", lt_list, fmt='%.6f')
        np.savetxt(result_path + "/l0_list.txt", l0_list, fmt='%.6f')


        plt.figure()
        plt.plot(mean_loss, alpha=0.6, label="loss")
        plt.plot(lt_list, alpha=0.6, label="lt_loss")
        plt.plot(l0_list, alpha=0.6, label="l0_loss")
        # plt.plot(pro_list[0], pro_list[1], alpha=0.6, label="proportion")
        # plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.xscale("log")
        plt.legend()
        plt.title('Training Loss')
        plt.savefig(result_path + "/loss_result.jpg")

        # plt.show()
        plt.close()

        plt.figure()
        plt.plot(epoch_list, kl_list, alpha=0.6, label="kl divergence")
        plt.legend()
        plt.savefig(result_path + "/kl.jpg")
        plt.close()

        plt.figure()
        plt.plot(self.check_point, kl_cp_list, "o-", alpha=0.6, label="kl divergence")
        plt.xscale("log")
        plt.legend()
        plt.savefig(result_path + "/kl_check_point.jpg")
        plt.close()

    def sample_and_evaluate(self, num_samples, epoch=args.epochs - 1, result_path=None):

        kl_list = []
        prob_list = []
        # t1 = time.time()
        if args.split_flat:
            result = []
            x_t_binary_init = []
            for _ in range(int(num_samples / 20000)):
                x_t_binary_part = tf.random.uniform([20000], minval=0, maxval=self.d, dtype=tf.int32)  # 随机初始化

                binary_list = generate_samples(20000, x_t_binary_part, self.N, self.num_timesteps, self.Theta,
                                               self.L, self.ansatz_connection)  # [num_samples, N]
                x_t_binary_init.append(x_t_binary_part)
                result.append(binary_list)  #
            binary_list = np.concatenate(result, axis=1)
            x_t_binary_init = tf.reshape(x_t_binary_init, [num_samples, -1])
        else:
            x_t_binary_init = tf.random.uniform([num_samples], minval=0, maxval=self.d, dtype=tf.int32)  # 随机初始化

            binary_list = generate_samples(num_samples, x_t_binary_init, self.N, self.num_timesteps, self.Theta,
                                           self.L, self.ansatz_connection)  # [num_samples, N]

        samples_np = binary_list[-1].astype(np.int32)  #

        samples_flat = samples_np.reshape(num_samples, -1)


        result_path = self.get_result_path(epoch=epoch, result_path=result_path)

        kl_0, frequencies_0 = self.draw(num_samples, samples_flat, epoch, t=0, result_path=result_path)


        print(f"generate {num_samples} samples, KL: {kl_0:.5f}")

        output_file_path = result_path + f"/output_samples/samples_epoch{epoch}_kl_{kl_0:.4f}.txt"
        if not args.fast_flat:
            with open(output_file_path, 'w') as f:
                for sample in samples_flat:

                    sample_str = ''.join(map(str, sample))  #
                    f.write(sample_str + '\n')  #


        x_t_binary_init = tf.bitwise.right_shift(tf.expand_dims(x_t_binary_init, axis=-1), tf.range(0, self.N)) & 1
        x_t_binary_init = tf.reshape(x_t_binary_init, [tf.shape(x_t_binary_init)[0], self.N])
        samples_flat = tf.cast(x_t_binary_init, dtype=tf.int32).numpy().reshape(num_samples, -1)
        # samples_flat = x_t_binary_init.astype(np.int32).reshape(num_samples, -1)
        kl_T, frequencies_T = self.draw(num_samples, samples_flat, epoch, t=args.num_timesteps, result_path=result_path)

        if not args.fast_flat:
            kl_list.append(kl_T)
            prob_list.append(frequencies_T)

            for i, t in enumerate(reversed(range(1, args.num_timesteps))):
                samples_flat = binary_list[i].reshape(num_samples, -1)
                kl_t, frequencies = self.draw(num_samples, samples_flat, epoch, t=t, result_path=result_path)
                kl_list.append(kl_t)
                prob_list.append(frequencies)
            kl_list.append(kl_0)
            prob_list.append(frequencies_0)
            np.savetxt(result_path + f"/bin_epoch_{epoch}/this_generation_process_kl_list.txt", kl_list, fmt='%.6f')
            np.savetxt(result_path + f"/bin_epoch_{epoch}/this_generation_process_prob_list.txt", prob_list, fmt='%.6f')
            with open(result_path + f"/model_params/Theta_{epoch}.pkl", "wb") as f:
                pickle.dump(self.Theta, f)

        return result_path, kl_0

    def draw(self, num_samples, samples_flat, epoch, t, result_path):
        decimal_values = [int(''.join(map(str, sample.flatten())), 2) for sample in samples_flat]

        frequency_count = Counter(decimal_values)

        all_possible_values = list(range(self.d))
        frequencies = [frequency_count.get(val, 0) / float(num_samples) for val in all_possible_values]
        frequencies_stand = self.gaussian_model.get_probs()

        kl_div = tf.keras.losses.kl_divergence(frequencies_stand, frequencies)

        self.draw_picture(frequencies, result_path, epoch, kl_div, t)
        return kl_div, frequencies

    def draw_picture(self, frequencies, result_path, epoch, kl, t, inf_flat=False):

        plt.figure(figsize=(10, 6))  #
        plt.bar(self.all_possible_values, self.frequencies_stand, alpha=0.8, label="target", color="skyblue")
        plt.bar(self.all_possible_values, frequencies, alpha=0.5, label="trained", color="orange")

        plt.margins(x=0.05)

        plt.xlabel('3x3 Binary Pattern (as Decimal)')
        plt.ylabel('Frequency')
        plt.title('Frequency of 3x3 Binary Patterns')
        plt.legend()
        plt.tight_layout()
        if not os.path.exists(result_path + f"/bin_epoch_{epoch}"):
            os.makedirs(result_path + f"/bin_epoch_{epoch}")
        if t == 0:
            plt.savefig(result_path + f"/Fre_epoch{epoch}_kl_{kl:.4f}_inf{str(inf_flat)}.jpg")
            plt.savefig(result_path + f"/bin_epoch_{epoch}" + f"/Frequency_t{t}_inf{str(inf_flat)}.jpg")
        else:
            plt.savefig(result_path + f"/bin_epoch_{epoch}" + f"/Frequency_t{t}_inf{str(inf_flat)}.jpg")
        plt.close()

        np.savetxt(result_path + f"/bin_epoch_{epoch}/frequencies_t{t}.txt", X=np.array(frequencies), fmt='%f')

        with open(result_path + f"/model_params/Theta_{epoch}.pkl", "wb") as f:
            pickle.dump(self.Theta, f)

    def get_result_path(self, epoch, result_path=None):
        if result_path is None:

            if args.sigma == 10000:
                sigma_type = "sigma_hybrid"
            else:
                sigma_type = f"sigma_{args.sigma}"

            ts = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            result_path = f'result_{self.N}qubit_gaussian/T{args.num_timesteps}/{args.ansatz_connection}/{sigma_type}/' \
                          f'L{args.layer}_seed{args.seed}'

            if not os.path.exists(result_path):
                os.makedirs(result_path)  # make dir
                os.makedirs(result_path + f"/bin_epoch_{epoch}")
                os.makedirs(result_path + f"/model_params")
                os.makedirs(result_path + f"/output_samples")

            argsDict = args.__dict__  # save setting
            with open(result_path + '/setting.txt', 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in argsDict.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')
        return result_path


if __name__ == '__main__':
    model = Model(args=args)
    model.forward()
