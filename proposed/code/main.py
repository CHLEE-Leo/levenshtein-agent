# %%
import os
import argparse
import tensorflow as tf
import numpy as np

def initialize_setting():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def get_params():
    parser = argparse.ArgumentParser(description='receive the parameters')

    # reward_function & Env_Model & RL_Agent 공통인자
    parser.add_argument('--my_seed', type = int, required = True)
    parser.add_argument('--task', type = str, required = True)                          # {'ST', 'DR'}, ST = Style Transfer', DR = Drug Reposition
    parser.add_argument('--train_for', type = str, required = True)                     # {'Env_Model', 'reward_function2', 'RL_Agent'}
    parser.add_argument('--num_epochs', type = int, required = True)
    parser.add_argument('--batch_size', type = int, required = True)
    parser.add_argument('--lr', type = float, required = True)
    parser.add_argument('--opt_schedule', type = str, required = False)                 #   {'CosDecay', 'CosDecayRe', 'ExpDecay', 'InvTimeDecay', 'None'}

    # RL_Agent 고유인자
    parser.add_argument('--len_buffer', type = int, required = False)                   #   {0, 1, 2}
    parser.add_argument('--eta', type = float, required = False)                        #   {0, 0.1, 0.005, 0.01, 1}
    parser.add_argument('--env_sampling', type = str, required = False)                 #   {'greedy', 'stochastic'}
    parser.add_argument('--reward', type = str, required = False)                       #   {'S', 'A', 'G', 'H'}, S = Sum, A = Arithmetic, G = Geometric, H = Harmonic
    parser.add_argument('--algo', type = str, required = False)                         #   {'PG', 'PPO'}
    parser.add_argument('--early_stop', type = str, required = False)                         #   {'yes', 'no'}
    # parser.add_argument('--random_search', type = str, required = False)                #   {'yes', 'no' }

    # parser.add_argument('--batch_size', type = int, required = True)
    # parser.add_argument('--num_epochs', type = int, required = True)
    # parser.add_argument('--lr', type = float, required = True)
    # parser.add_argument('--rnn_cell', type = str, required = True)
    # parser.add_argument('--head_type', type = str, required = True)
    # parser.add_argument('--cla_dropout', type = str, required = True)
    # parser.add_argument('--cla_regularization', type = str, required = True)
    global args
    args = parser.parse_args()
    return args

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    # GPU setting
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    initialize_setting()

    # 글로벌 파라미터 가져오기
    args = get_params()

    # 시드 고정
    seed_everything(args.my_seed)

    # 훈련 실행
    from train import *