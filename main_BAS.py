import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import argparse
import pickle
from collections import Counter
from BAS_Gaussain_mode import *
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

parser.add_argument('--N', type=int, default=4)

parser.add_argument('--num_timesteps', type=int, default=30)

parser.add_argument('--sigma', type=float, default=10000, help="10000: hybrid")

parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--ansatz_connection', type=str, default="all_to_all", help="all_to_all | star | chain")

parser.add_argument('--epochs', type=int, default=6000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_samples', type=int, default=10000)

parser.add_argument('--lr_decay_type', type=str, default="cos")        # cos exp none
parser.add_argument('--initial_lr', type=float, default=0.001)  #   cos
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--min_lr', type=float, default=0.0001)       # cos

parser.add_argument('--decay_steps', type=int, default=3000)    # cos

parser.add_argument('--fast_flat', type=bool, default=False)
parser.add_argument('--split_flat', type=bool, default=False)
parser.add_argument('--check_point', type=bool, default=True)


args = parser.parse_args()

# assert args.N == 4 or args.N == 9, ValueError("only 4 or 9 BAS problem")



np.random.seed(args.seed)
tf.random.set_seed(args.seed)





if int(args.sigma) == 10000:
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
        # self.gaussian_model = Discrete_Gaussian(batch_size=args.batch_size, n_bits=self.N, num_dataset=50000)
        if args.N == 4:
            self.frequencies_stand = [1 / 6., 0, 0, 1 / 6., 0, 1 / 6., 0, 0, 0, 0, 1 / 6., 0, 1 / 6., 0, 0, 1 / 6.]
            self.indices = [0, 3, 5, 10, 12, 15]
            self.dataset = np.array(BAS_2x2, dtype=np.int32)

        elif args.N == 6:
            self.frequencies_stand = np.zeros(shape=[2 ** args.N], dtype=np.float32)
            self.indices = [56, 7, 36, 18, 9, 54, 27, 45, 63, 0]
            self.frequencies_stand[self.indices] = 1. / 10
            self.dataset = np.array(BAS_2x3, dtype=np.int32)
        elif args.N == 8:
            self.frequencies_stand = np.zeros(shape=[2 ** args.N], dtype=np.float32)
            self.indices = [240, 15, 136, 68, 34, 17, 204, 170, 153, 102, 85, 51, 238, 221, 187, 119, 255, 0]
            self.frequencies_stand[self.indices] = 1. / len(self.indices)
            self.dataset = np.array(BAS_2x4, dtype=np.int32)
        elif args.N == 9:
            self.frequencies_stand = np.zeros(shape=[2 ** args.N], dtype=np.float32)
            self.indices = [448, 56, 7, 504, 63, 455, 292, 146, 73, 438, 219, 365, 511, 0]
            self.frequencies_stand[self.indices] = 1. / 14.
            self.dataset = np.array(BAS_3x3, dtype=np.int32)
        elif args.N == 10:
            self.frequencies_stand = np.zeros(shape=[2 ** args.N], dtype=np.float32)
            self.indices = [992, 31, 528, 264, 132, 66, 33, 792, 660, 594, 561, 396, 330, 297, 198, 165, 99, 924, 858,
                            825, 726, 693, 627, 462, 429, 363, 231, 990, 957, 891, 759, 495, 1023, 0]
            self.frequencies_stand[self.indices] = 1. / len(self.indices)
            self.dataset = np.array(BAS_2x5, dtype=np.int32)
        elif args.N == 12:
            self.frequencies_stand = np.zeros(shape=[2 ** args.N], dtype=np.float32)
            self.indices = [3840, 240, 15, 4080, 255, 3855, 2184, 1092, 546, 273, 3276, 2730, 2457, 1638,
                            1365, 819, 3822, 3549, 3003, 1911, 4095, 0]
            self.frequencies_stand[self.indices] = 1. / len(self.indices)
            self.dataset = np.array(BAS_3x4, dtype=np.int32)
        elif args.N == 16:
            self.frequencies_stand = np.zeros(shape=[2 ** args.N], dtype=np.float32)
            self.indices = [61440, 3840, 240, 15, 65280, 61680, 61455, 4080, 3855, 255, 65520, 65295, 61695, 4095,
                            34952, 17476, 8738, 4369, 52428, 43690, 39321, 26214, 21845, 13107, 61166, 57021, 48059,
                            30583, 65535, 0]
            self.frequencies_stand[self.indices] = 1. / len(self.indices)
            self.dataset = np.array(BAS_4x4, dtype=np.int32)

        self.all_possible_values = list(range(self.d))

        if args.check_point:
            self.check_point = [0, 1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000,
                                1568, 2446, 3832, 5999]
        else:
            self.check_point = []
        # if args.N == 4:
        #     self.dataset = np.array(BAS_2x2, dtype=np.int32)
        # else:
        #     self.dataset = np.array(BAS_3x3, dtype=np.int32)

        self.cut_index = [[i for i in range(args.N) if i != n] for n in range(args.N)]

        self.binary_labels = [bin(i)[2:].zfill(args.N) for i in self.all_possible_values]

    def forward(self):
        mean_loss = []

        result_path = None
        epoch_list = []
        lt_list, l0_list = [], []
        kl_list = []
        pro_list = []
        kl_cp_list = []

        for epoch in range(self.args.epochs):
            t = tf.random.uniform(shape=[self.batch_size], minval=2, maxval=self.num_timesteps + 1, dtype=tf.int32)

            x_batch = []
            for _ in range(self.batch_size):
                pattern = self.dataset[np.random.randint(0, self.dataset.shape[0])]
                x_batch.append(pattern)
            x_batch = np.array(x_batch, dtype=np.int32)
            x_batch = x_batch.reshape(self.batch_size, -1)    # (self.batch_size, 10)

            loss, lt, l0 = train_step(x_batch, t, self.N, self.d, self.alpha_bar,
                                      self.alpha_t, self.Theta, self.opt, self.cm_state, self.num_timesteps,
                                      self.batch_size, mmd_loss, self.L, self.ansatz_connection)
            lt_list.append(lt.numpy())
            l0_list.append(l0.numpy())
            mean_loss.append(loss.numpy())

            if (epoch + 1) % 500 == 0 or epoch == args.epochs - 1 or epoch == 0 or epoch in self.check_point :
                print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}, lt:{lt.numpy():.4f}, l0:{l0.numpy():.4f}")
                result_path, kl = self.sample_and_evaluate(num_samples=args.num_samples, epoch=epoch, result_path=result_path)

                if (epoch + 1) % 500 == 0 or epoch == args.epochs - 1 or epoch == 0:
                    epoch_list.append(epoch)
                    kl_list.append(kl)
                if epoch in self.check_point:
                    kl_cp_list.append(kl)

        np.savetxt(result_path + "/epoch_list.txt", epoch_list, fmt='%d')
        np.savetxt(result_path + "/pro_list.txt", pro_list, fmt='%.4f')
        np.savetxt(result_path + "/kl_list.txt", kl_list, fmt='%.5f')
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

        plt.legend()
        plt.title('Training Loss')
        plt.savefig(result_path + "/loss_result.jpg")


        plt.close()

        plt.figure()
        plt.plot(epoch_list, kl_list, alpha=0.6, label="kl divergence")
        plt.legend()
        plt.savefig(result_path + "/kl and acc.jpg")
        plt.close()

        plt.figure()
        plt.plot(self.check_point, kl_cp_list, "o-", alpha=0.6, label="kl divergence")
        # plt.plot(self.check_point, pro_cp_list, alpha=0.6, label="acc")
        plt.xscale("log")
        plt.legend()
        plt.savefig(result_path + "/kl_check_point.jpg")
        plt.close()

    def sample_and_evaluate(self, num_samples, epoch=args.epochs - 1, result_path=None):

        kl_list = []
        prob_list = []
        if args.split_flat:
            result = []
            x_t_binary_init = []
            for _ in range(int(num_samples / 20000)):
                x_t_binary_part = tf.random.uniform([20000], minval=0, maxval=self.d, dtype=tf.int32)  #

                binary_list = generate_samples(20000, x_t_binary_part, self.N, self.num_timesteps, self.Theta,
                                           self.L, self.ansatz_connection)  # [num_samples, N]

                result.append(binary_list)  # [11, 30, 2000, 9]
                x_t_binary_init.append(x_t_binary_part)
            binary_list = np.concatenate(result, axis=1)
            x_t_binary_init = tf.reshape(x_t_binary_init, [num_samples, -1])
        else:
            x_t_binary_init = tf.random.uniform([num_samples], minval=0, maxval=self.d, dtype=tf.int32)  #

            binary_list = generate_samples(num_samples, x_t_binary_init, self.N, self.num_timesteps, self.Theta,
                                           self.L, self.ansatz_connection)  # [num_samples, N]
        # binary_list = binary_list.numpy()
        samples_np = binary_list[-1].astype(np.int32)  #

        samples_flat = samples_np.reshape(num_samples, -1)

        patterns_flat = tf.reshape(self.dataset,(self.dataset.shape[0], -1))

        matches = np.any(np.all(samples_flat[:, None, :] == patterns_flat[None, :, :], axis=2), axis=1)

        match_count = np.sum(matches)

        proportion = match_count / num_samples
        result_path = self.get_result_path(epoch=epoch, result_path=result_path)
        kl_0, frequencies_0 = self.draw(num_samples, samples_flat, epoch, t=0, result_path=result_path, proportion=proportion)
        # print(f"{kl_div:.5f}")

        print(f" generate {num_samples} samples, Kl: {kl_0:.5f}")

        output_file_path = result_path + f"/output_samples/samples_epoch{epoch}_kl_{kl_0:.4f}.txt"
        if not args.fast_flat:
            with open(output_file_path, 'w') as f:
                for sample in samples_flat:

                    sample_str = ''.join(map(str, sample))
                    f.write(sample_str + '\n')

        x_t_binary_init = tf.bitwise.right_shift(tf.expand_dims(x_t_binary_init, axis=-1), tf.range(0, self.N)) & 1
        x_t_binary_init = tf.reshape(x_t_binary_init, [tf.shape(x_t_binary_init)[0], self.N])
        samples_flat = tf.cast(x_t_binary_init, dtype=tf.int32).numpy().reshape(num_samples, -1)

        kl_T, frequencies_T = self.draw(num_samples, samples_flat, epoch, t=args.num_timesteps, result_path=result_path, proportion=proportion)

        if not args.fast_flat:
            kl_list.append(kl_T)
            prob_list.append(frequencies_T)

            for i, t in enumerate(reversed(range(1, args.num_timesteps))):
                samples_flat = binary_list[i].reshape(num_samples, -1)
                kl_t, frequencies = self.draw(num_samples, samples_flat, epoch, t=t, result_path=result_path,proportion=proportion)
                kl_list.append(kl_t)
                prob_list.append(frequencies)
            kl_list.append(kl_0)
            prob_list.append(frequencies_0)
            np.savetxt(result_path + f"/bin_epoch_{epoch}/this_generation_process_kl_list.txt", kl_list, fmt='%.6f')
            np.savetxt(result_path + f"/bin_epoch_{epoch}/this_generation_process_prob_list.txt", prob_list, fmt='%.6f')
            with open(result_path + f"/model_params/Theta_{epoch}.pkl", "wb") as f:
                pickle.dump(self.Theta, f)

        return result_path, kl_0

    def draw(self, num_samples, samples_flat, epoch, t, result_path, proportion=0.1):
        decimal_values = [int(''.join(map(str, sample.flatten())), 2) for sample in samples_flat]

        frequency_count = Counter(decimal_values)

        all_possible_values = list(range(self.d))
        frequencies = [frequency_count.get(val, 0) / float(num_samples) for val in all_possible_values]

        kl_div = tf.keras.losses.kl_divergence(self.frequencies_stand, frequencies)


        # prob_list[-1], result_path, epoch, acc_list[-1], kl[-1], t=0
        self.draw_picture(frequencies, result_path, epoch, proportion, kl_div, t)
        return kl_div, frequencies

    def draw_picture(self, frequencies, result_path, epoch, proportion, kl_div, t, inf_flat=False):

        plt.bar(self.all_possible_values, self.frequencies_stand, alpha=0.6, label="BAS", color="skyblue")
        plt.bar(self.all_possible_values, frequencies, alpha=0.6, label="trained", color="orange")


        selected_labels = [self.binary_labels[i] for i in self.indices]
        plt.xticks(self.indices, selected_labels, rotation=45, ha='right', fontsize=8)  #
        plt.margins(x=0.05)

        plt.xlabel(f'{self.N}x{self.N} Binary Pattern (as Decimal)')
        plt.ylabel('Frequency')
        plt.title('Frequency of 3x3 Binary Patterns')
        plt.legend()
        plt.tight_layout()
        if not os.path.exists(result_path + f"/bin_epoch_{epoch}"):
            os.makedirs(result_path + f"/bin_epoch_{epoch}")
        if t==0:
            plt.savefig(result_path + f"/Fre_epoch{epoch}_pro_{proportion:.4f}_kl_{kl_div:.4f}_inf{str(inf_flat)}.jpg",dpi=300)
            plt.savefig(result_path + f"/bin_epoch_{epoch}" + f"/Frequency_t{t}_inf{str(inf_flat)}.jpg",dpi=300)
        else:
            plt.savefig(result_path + f"/bin_epoch_{epoch}" + f"/Frequency_t{t}_inf{str(inf_flat)}.jpg",dpi=300)
        plt.close()

        np.savetxt(result_path + f"/bin_epoch_{epoch}/frequencies_t{t}_inf_flat{str(inf_flat)}.txt", X=np.array(frequencies), fmt='%f')

        with open(result_path + f"/model_params/Theta_{epoch}.pkl", "wb") as f:
            pickle.dump(self.Theta, f)

    def get_result_path(self, epoch, result_path=None):
        if result_path is None:
            if args.sigma == 10000:
                sigma_type = "sigma_hybrid"
            else:
                sigma_type = f"sigma_{args.sigma}"
            ts = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            result_path = f'result_{self.N}qubit_BAS/T{args.num_timesteps}/{args.ansatz_connection}/{sigma_type}/' \
                          f'L{args.layer}_seed{args.seed}_{ts}'

            if not os.path.exists(result_path):
                os.makedirs(result_path)  # make dir
                os.makedirs(result_path + f"/bin_epoch_{epoch}")
                os.makedirs(result_path + f"/model_params")
                os.makedirs(result_path + f"/output_samples")

            argsDict = args.__dict__  # save setting
            with open(result_path + '/setting.txt', 'w') as f:
                f.writelines("Time: " + ts + '\n')
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in argsDict.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')
        return result_path



if __name__ == '__main__':

    model = Model(args=args)
    model.forward()
