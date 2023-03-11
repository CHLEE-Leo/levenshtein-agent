# %%
import os
from sys import getsizeof, exit
from tkinter import E
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle, json
import copy
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from model import AETransformer, Reward_Function, Mask_Generator, LEVAgent
from main import get_params
from utils import *
import gc
import time

# 글로벌 파라미터 가져오기
args = get_params()
train_for = args.train_for
learning_rate = args.lr
OptSchedule = args.opt_schedule
target_task = args.task    # ST : Style Transfer, DR : Drug Repositioning
batch_size = args.batch_size
# target_task = input('task (i.e., ST vs. DR) : ')    # ST : Style Transfer, DR : Drug Repositioning

'''
kwargs 파라미터들은 학습 및 추론을 위해 manually 세팅해주어야 함.

만약 특정 controller (= classifier)에 대해 generator를 제어하고 싶다면, 
hyper_param_dir + '/controller' 주소 아래의 하이퍼 파라미터 조건을 보고
그에 맞추어 아래 kwargs 리스트를 수정한 뒤 run 할 것.
'''


'''
target_task가 ST일 때와 DR일 떄의 차이점

target_task = 'DR' 일 때에는
1) reward_model 훈련시에는 train_input_sequence가 실은 output_sequence(train).npy임
2) Env_Model 및 RL_Agent 훈련시에는
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------- #

# 파라미터 설정, 데이터 로드, 각종 메트릭 및 최적화 알고리즘 정의
# -- (1) reward_function ----------------------------------------------------------------------------------------------------------------- #
# if train_for == 'reward_function1' or train_for == 'reward_function2':
if train_for == 'reward_function2':

    if target_task == 'ST':

        # 로드 및 세이브 주소 지정
        final_dir = '/text-style-transfer'
        load_data_dir = '/home/messy92/Leo/NAS_folder/ICML23/prep_data' + final_dir
        hyper_param_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters' + final_dir
        save_weight_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights' + final_dir
        save_result_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/results'

        # 학습 데이터 로드
        train_input_sequence = np.load(load_data_dir + '/input_sequence(train).npy')
        eos_idx = indexing_eos_token(train_input_sequence)
        train_input_sequence = train_input_sequence[np.where(eos_idx >= 4)[0], :]           # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        train_attribute = np.load(load_data_dir + '/attribute(train).npy')
        train_attribute = train_attribute[np.where(eos_idx >= 4)[0]]                          # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

        # 검증 데이터 로드
        val_input_sequence = np.load(load_data_dir + '/input_sequence(val).npy')
        eos_idx = indexing_eos_token(val_input_sequence)
        val_input_sequence = val_input_sequence[np.where(eos_idx >= 4)[0], :]               # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        val_attribute = np.load(load_data_dir + '/attribute(val).npy')
        val_attribute = val_attribute[np.where(eos_idx >= 4)[0]]                              # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)


    elif target_task == 'DR':

        # 로드 및 세이브 주소 지정
        final_dir = '/drug-discovery'
        load_data_dir = '/home/messy92/Leo/NAS_folder/ICML23/prep_data' + final_dir
        hyper_param_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters' + final_dir
        save_weight_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights' + final_dir
        save_result_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/results'

        # 학습 데이터 로드
        train_input_sequence = np.load(load_data_dir + '/output_sequence(train).npy')           # ST와 달리 DR에서 reward_model의 훈련에 사용되는 train_input_sequence는 output_sequence(train).npy임.
        eos_idx = indexing_eos_token(train_input_sequence)
        train_input_sequence = train_input_sequence[np.where(eos_idx >= 4)[0], :]               # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        train_attribute = np.load(load_data_dir + '/attribute(train).npy')
        train_attribute = train_attribute[np.where(eos_idx >= 4)[0]]                            # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

        # 검증 데이터 로드
        val_input_sequence = np.load(load_data_dir + '/output_sequence(val).npy')               # ST와 달리 DR에서 reward_model의 훈련에 사용되는 val_input_sequence는 output_sequence(val).npy임.
        eos_idx = indexing_eos_token(val_input_sequence)
        val_input_sequence = val_input_sequence[np.where(eos_idx >= 4)[0], :]                   # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        val_attribute = np.load(load_data_dir + '/attribute(val).npy')
        val_attribute = val_attribute[np.where(eos_idx >= 4)[0]]                                # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)


    # 단어 토큰 가져오기
    with open(load_data_dir + '' + '/token_dict.pickle', 'rb') as f:
        token_dict = pickle.load(f)
    special_token_list = ['[pad]', '[mask]']
    edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
    add_token_list = special_token_list + edit_token_list
    token_dict = add_token_list_in_dict(add_token_list, token_dict)
    action_set = list(token_dict.values())[-len(edit_token_list):]

    # 파라미터 설정
    # if train_for == 'reward_function1':
    #     model_name = 'lik_reward_function'
    # elif train_for == 'reward_function2':
    #     model_name = 'attr_reward_function'
    model_name = 'attr_reward_function'

    kwargs = {
        'target_task' : target_task,
        'model_name' : model_name,
        'batch_size' : batch_size,
        'lr' : learning_rate,
        'num_layers_enc' : 4,
        'd_model' : 0,
        'd_model_enc' : 256,
        'num_heads' : 0,
        'num_heads_enc' : 8,
        'd_ff' : 512,
        'stack': 'mlp',
        'dropout_rate' : 0.1,
        'vocab_size' : len(token_dict),
        'num_epochs' : args.num_epochs,
        'action_space' : action_set
    }

    print(os.getcwd())
    task_path = '_' + target_task
    batch_size_path = '_' + str(batch_size)
    stack_path = '_' + str(kwargs['stack'])
    epoch_path = '_' + str(kwargs['num_epochs'])

    if os.path.isfile(hyper_param_dir + '/Reward_Function' + '/kwargs' + '_' + model_name + task_path + epoch_path + batch_size_path) == False:
        with open(hyper_param_dir + '/Reward_Function' + '/kwargs' + '_' + model_name + task_path + epoch_path + batch_size_path, 'w') as f:
            json.dump(kwargs, f, indent = '\t')

    # 최적화 알고리즘, 손실함수 및 정확도 함수
    optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])    

    # 보상함수 손실 함수
    sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    def re_loss_function(real, pred):
        losses = sparse_categorical_cross_entropy(real, pred)
        return losses

    # 보상함수 정확도 함수
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_categorical_accuracy')
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_categorical_accuracy')
    def re_accuracy_function(real, pred, mode):
        
        if mode == 'train':
            train_acc_metric.update_state(real, pred)
            mean_acc = train_acc_metric.result()
        elif mode == 'test':
            val_acc_metric.update_state(real, pred)
            mean_acc = val_acc_metric.result()

        return tf.cast(mean_acc, dtype = tf.float32)

    @tf.function
    def train_step(data, model):
        x, y = data
        pad_mask, _, _ = model.mask_generator(x, x)

        with tf.GradientTape() as tape:
            # 예측
            y_hat = model(x, pad_mask, training = True)

            # 손실 및 정확도 계산
            losses = re_loss_function(y, y_hat)

            # 최종 손실
            total_losses = losses

        # 최적화
        gradients = tape.gradient(total_losses, model.trainable_variables)
        optimizers.apply_gradients(zip(gradients, model.trainable_variables))
        accuracies = re_accuracy_function(y, y_hat, mode = 'train')

        return losses, accuracies

    @tf.function
    def test_step(data, model):
        x, y = data
        pad_mask, _, _ = model.mask_generator(x, x)

        # 예측
        y_hat = model(x, pad_mask, training = False)

        # 손실 및 정확도 계산
        losses = re_loss_function(y, y_hat)
        accuracies = re_accuracy_function(y, y_hat, mode = 'test')

        return losses, accuracies

# -- (2) AE_LM & Mask_LM & GPT & RL_Agent ----------------------------------------------------------------------------------------------------------------- #
elif 'Env_Model' or train_for == 'RL_Agent':

    if target_task == 'ST':

        # 로드 및 세이브 주소 지정
        final_dir = '/text-style-transfer'
        load_data_dir = '/home/messy92/Leo/NAS_folder/ICML23/prep_data' + final_dir
        hyper_param_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters' + final_dir
        save_weight_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights' + final_dir
        save_result_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/results'

        # 학습 데이터 로드
        train_input_sequence = np.load(load_data_dir + '/input_sequence(train).npy')
        train_attribute = np.load(load_data_dir + '/attribute(train).npy')
        eos_idx = indexing_eos_token(train_input_sequence)
        train_input_sequence = train_input_sequence[np.where(eos_idx >= 4)[0], :]           # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        train_attribute = train_attribute[np.where(eos_idx >= 4)[0]]                        # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        
        # 검증 데이터 로드
        val_input_sequence = np.load(load_data_dir + '/input_sequence(val).npy')
        val_attribute = np.load(load_data_dir + '/attribute(val).npy')
        eos_idx = indexing_eos_token(val_input_sequence)
        val_input_sequence = val_input_sequence[np.where(eos_idx >= 4)[0], :]               # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        val_attribute = val_attribute[np.where(eos_idx >= 4)[0]]                            # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

        # 실험 데이터 로드
        test_input_sequence = np.load(load_data_dir + '/input_sequence(test).npy')
        test_attribute = np.load(load_data_dir + '/attribute(test).npy')
        eos_idx = indexing_eos_token(test_input_sequence)
        test_input_sequence = test_input_sequence[np.where(eos_idx >= 4)[0], :]             # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        test_attribute = test_attribute[np.where(eos_idx >= 4)[0]]                          # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

        # 마스킹 파라미터인 mean_num_mask (= 평균 마스크 갯수) 정의
        mean_num_mask = get_masking_param(train_input_sequence)                             # train_input_sequence인 것에 주의

    elif target_task == 'DR':

        # 로드 및 세이브 주소 지정
        final_dir = '/drug-discovery'
        load_data_dir = '/home/messy92/Leo/NAS_folder/ICML23/prep_data' + final_dir
        hyper_param_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters' + final_dir
        save_weight_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights' + final_dir
        save_result_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/results'

        '''
        Env_Model 및 RL_Agent 훈련에 사용되는 train_output_sequence가 곧 train_input_sequence (= input_sequence(train).npy)인 ST와 달리,
        DR에서는 train_output_sequence = output_sequence(train).npy 임.

        Env_Model 및 RL_Agent 훈련에 사용되는 val_output_sequence가 곧 val_input_sequence (= input_sequence(val).npy)인 ST와 달리,
        DR에서는 train_output_sequence = output_sequence(val).npy 임.
        '''

        # 학습 데이터 로드
        train_input_sequence = np.load(load_data_dir + '/input_sequence(train).npy')
        train_output_sequence = np.load(load_data_dir + '/output_sequence(train).npy')
        train_attribute = np.load(load_data_dir + '/attribute(train).npy')
        eos_idx = indexing_eos_token(train_input_sequence)
        train_input_sequence = train_input_sequence[np.where(eos_idx >= 4)[0], :]               # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        train_output_sequence = train_output_sequence[np.where(eos_idx >= 4)[0], :]             # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        train_attribute = train_attribute[np.where(eos_idx >= 4)[0]]                            # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        
        # 검증 데이터 로드
        val_input_sequence = np.load(load_data_dir + '/input_sequence(val).npy')
        val_output_sequence = np.load(load_data_dir + '/output_sequence(val).npy')
        val_attribute = np.load(load_data_dir + '/attribute(val).npy')
        eos_idx = indexing_eos_token(val_input_sequence)
        val_input_sequence = val_input_sequence[np.where(eos_idx >= 4)[0], :]                   # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        val_output_sequence = val_output_sequence[np.where(eos_idx >= 4)[0], :]             # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
        val_attribute = val_attribute[np.where(eos_idx >= 4)[0]]                                # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

        # 마스킹 파라미터인 mean_num_mask (= 평균 마스크 갯수) 정의
        mean_num_mask = get_masking_param(train_output_sequence)                                # train_output_sequence인 것에 주의

    # 토큰 사전 가져오기
    with open(load_data_dir + '' + '/token_dict.pickle', 'rb') as f:
        token_dict = pickle.load(f)
    special_token_list = ['[pad]', '[mask]']
    reward_class_token_list = ['[' + 'R_' + str(reward_class) + ']' for reward_class in range(len(np.unique(train_attribute)))]
    edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
    add_token_list = special_token_list + reward_class_token_list + edit_token_list
    token_dict = add_token_list_in_dict(add_token_list, token_dict)
    action_set = list(token_dict.values())[-len(edit_token_list):]


    # 파라미터 설정         
    if train_for == 'Env_Model':
        # model_name = input('decoder (BART vs. NAR) : ')
        # model_name = 'BART'
        model_name = 'NAR'

    elif train_for == 'RL_Agent':
        model_name = 'LEVA'

    elif train_for == 'AR_LM':
        model_name = 'GPT'

    # Env_Model (= BART) 사용자 정의 파라미터
    if train_for == 'Env_Model':
        kwargs = {
            'target_task' : target_task,
            'model_name' : model_name,
            'batch_size' : batch_size,
            'lr' : learning_rate,
            'num_layers_enc' : 2,
            'num_layers_dec' : 2,
            'd_model' : 0,
            'd_model_enc' : 512,
            'd_model_dec' : 512,
            'num_heads' : 0,
            'num_heads_enc' : 4,
            'num_heads_dec' : 4,
            'd_ff' : 1024,
            'stack': 'rnn',
            'dropout_rate' : 0.1,
            'vocab_size' : len(token_dict),
            'num_epochs' : args.num_epochs,
            'action_space' : action_set
        }

    # RL_Agent (= LEVA) 사용자 정의 파라미터
    elif train_for == 'RL_Agent':
        kwargs = {
            'target_task' : target_task,
            'model_name' : model_name,
            'batch_size' : batch_size,
            'lr' : learning_rate,
            'num_layers_enc' : 1,
            'num_layers_dec' : 1,
            'd_model' : 0,
            'd_model_enc' : 512,
            'd_model_dec' : 512,
            'num_heads' : 0,
            'num_heads_enc' : 2,
            'num_heads_dec' : 2,
            'd_ff' : 1024,
            'stack': 'rnn',
            'dropout_rate' : 0.1,
            'vocab_size' : len(token_dict),
            'num_epochs' : args.num_epochs,
            'action_space' : action_set,
            'len_buffer' : args.len_buffer,
            'eta' : args.eta,
            'env_sampling' : args.env_sampling,
            'reward' : args.reward,
            'algo' : args.algo,
            'early_stop' : args.early_stop
        }

    # 저장 디렉토리 설정 및 파라미터 딕셔너리를 json 파일로 저장
    print(os.getcwd())

    # 훈련함수
    if train_for == 'Env_Model':
        task_path = '_' + target_task
        batch_size_path = '_' + str(batch_size)
        stack_path = '_' + str(kwargs['stack'])
        epoch_path = '_' + str(kwargs['num_epochs'])

        if os.path.isfile(hyper_param_dir + '/' + model_name + '/kwargs' + '_' + model_name + task_path + epoch_path + batch_size_path) == False:
            with open(hyper_param_dir + '/' + model_name + '/kwargs' + '_' + model_name + task_path + epoch_path + batch_size_path, 'w') as f:
                json.dump(kwargs, f, indent = '\t')
 
        '''
        AutoEncoding Transformer / BART / BART 훈련 및 검증 함수
        '''

        # 최적화 알고리즘, 손실함수 및 정확도 함수
        optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])

        @tf.function
        def train_step(data, model):

            inputs, outputs, targets = data

            with tf.GradientTape() as tape1:

                # 예측
                enc_outputs, _, dec_outputs, _ = model((inputs, outputs), training = True)

                # 손실 및 정확도 계산
                losses, _ = aet_loss_function(targets, dec_outputs)
                accuracies = aet_accuracy_function(targets, dec_outputs)

                # 최종 손실
                total_losses = losses

            # 최적화
            gradients = tape1.gradient(total_losses, model.trainable_variables)
            optimizers.apply_gradients(zip(gradients, model.trainable_variables))

            return losses, accuracies

        @tf.function
        def test_step(data, model):

            inputs, outputs, targets = data

            # 예측
            enc_outputs, _, dec_outputs, _ = model((inputs, outputs), training = False)

            # 손실 및 정확도 계산
            losses, _ = aet_loss_function(targets, dec_outputs)
            accuracies = aet_accuracy_function(targets, dec_outputs)

            return losses, accuracies

    elif train_for == 'RL_Agent':
        task_path = '_' + target_task
        batch_size_path = '_' + str(batch_size)
        epoch_path = '_' + str(kwargs['num_epochs'])

        if os.path.isfile(hyper_param_dir + '/' + model_name + '/kwargs' + '_' + model_name + task_path + epoch_path + batch_size_path) == False:
            with open(hyper_param_dir + '/' + model_name + '/kwargs' + '_' + model_name + task_path + epoch_path + batch_size_path, 'w') as f:
                json.dump(kwargs, f, indent = '\t')

        '''
        강화학습 훈련 및 검증 함수
        '''
        # 최적화 알고리즘, 손실함수 및 정확도 함수
        if OptSchedule == 'CosDecay':
            cos_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=kwargs['lr'], decay_steps=30, alpha=1e-5)                                       # initial_learning_rate = 초기 lr
                                                                                                                                                                        # decay_steps = decay 횟수
                                                                                                                                                                        # alpha = 최소 lr (= alpha * initial_learning_rate)
            optimizers = tf.keras.optimizers.Adam(learning_rate = cos_decay)

        elif OptSchedule == 'CosDecayRe':
            cos_decay_re = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=kwargs['lr'], first_decay_steps=10, t_mul=1, m_mul=0.9, alpha=1e-5)      # first_decay_steps = 초기 decay 횟수
                                                                                                                                                                            # t_mul = restart 시 decay_steps 증가 배수
                                                                                                                                                                            # m_mul = restart 초기화 비율
            optimizers = tf.keras.optimizers.Adam(learning_rate = cos_decay_re)

        elif OptSchedule == 'ExpDecay':
            exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=kwargs['lr'], decay_steps=30, decay_rate=0.9)       # decay_steps = 초기 decay 횟수
                                                                                                                                                # decay_rate = decay 비율
            optimizers = tf.keras.optimizers.Adam(learning_rate = exp_decay)

        elif OptSchedule == 'InvTimeDecay':
            inv_time_decay = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=kwargs['lr'], decay_steps=30, decay_rate=0.9, staircase=False, name=None)          # decay_steps = 초기 decay 횟수
                                                                                                                                                                                    # decay_rate = decay 비율
            optimizers = tf.keras.optimizers.Adam(learning_rate = inv_time_decay)

        else:   # == 'None'
            optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])


        @tf.function
        def agent_train_step(data, lev_agent, enc_pad_mask, eta, algo):
            
            if algo == 'PG':
                inputs, _, agent_actions, total_rewards = data

                # Train encoder & decoder
                with tf.GradientTape() as tape:

                    '''
                    편집 계획 (edit_plans) 생성
                    '''
                    agent_outputs, _ = lev_agent(inputs, enc_pad_mask)

                    '''
                    손실 계산
                    '''
                    # 레반슈타인 손실 계산 (LEVA에게 Accuracy를 정의할 수 있는 기준은 없음.)
                    lev_losses = lev_loss_function(agent_actions, agent_outputs)         # lev_losses : (batch_size, )                

                    # 규제항 계산
                    uniform_dist = tf.nn.softmax(tf.random.uniform(shape = tf.shape(agent_outputs)), axis = -1)
                    action_dist = tf.nn.softmax(agent_outputs, axis = -1) + 1e-10                                       # KL[p||q]에서, q가 0이 되는 경우가 존재하면 log(p/q)가 음의 무한대가 됨. 따라서, 이 경우가 발생하는 것을 방지하고자 q에 1e-10을 더해주기
                    div_score = tf.reduce_mean(KL_divergence(uniform_dist, action_dist), axis = -1)

                    # 규제된 레반슈타인 손실 계산
                    regularized_lev_losses = lev_losses + eta * div_score
                    
                    # 최종 손실
                    total_losses = tf.matmul(total_rewards[tf.newaxis, :], regularized_lev_losses[:, tf.newaxis])

            elif algo == 'PPO':
                inputs, old_agent_outputs, old_agent_actions, total_rewards = data

                # Train encoder & decoder
                with tf.GradientTape() as tape:

                    '''
                    편집 계획 (edit_plans) 생성
                    '''
                    agent_outputs, _ = lev_agent(inputs, enc_pad_mask)

                    '''
                    손실 계산
                    '''
                    # 레반슈타인 손실 계산 (LEVA에게 Accuracy를 정의할 수 있는 기준은 없음.)
                    lev_losses = lev_loss_function(old_agent_actions, agent_outputs)                # lev_losses : (batch_size, )                

                    # 정책비 계산 (ratio)
                    cur_pi = tf.nn.softmax(agent_outputs, axis = -1)
                    old_pi = tf.nn.softmax(old_agent_outputs, axis = -1)
                    policy_ratio = cur_pi / old_pi
                    clipped_importance = tf.clip_by_value(policy_ratio, clip_value_min=1-0.15, clip_value_max=1+0.15)
                    mean_clipped_importance = tf.reduce_mean(tf.reduce_mean(clipped_importance, axis = -1), axis = -1)

                    # 규제항 계산
                    uniform_dist = tf.nn.softmax(tf.random.uniform(shape = tf.shape(agent_outputs)), axis = -1)
                    action_dist = tf.nn.softmax(agent_outputs, axis = -1) + 1e-10                                       # KL[p||q]에서, q가 0이 되는 경우가 존재하면 log(p/q)가 음의 무한대가 됨. 따라서, 이 경우가 발생하는 것을 방지하고자 q에 1e-10을 더해주기
                    div_score = tf.reduce_mean(KL_divergence(uniform_dist, action_dist), axis = -1)

                    # 규제된 레반슈타인 손실 계산
                    regularized_lev_losses = lev_losses + eta * div_score
                    
                    # # 최종 손실
                    # total_losses = tf.matmul(total_rewards[tf.newaxis, :], regularized_lev_losses[:, tf.newaxis])

                    # 최종 손실
                    clipped_importance_rewards = tf.math.multiply(total_rewards, mean_clipped_importance)
                    total_losses = tf.matmul(clipped_importance_rewards[tf.newaxis, :], regularized_lev_losses[:, tf.newaxis])


            # 최적화
            gradients = tape.gradient(total_losses, lev_agent.trainable_variables)
            optimizers.apply_gradients(zip(gradients, lev_agent.trainable_variables))

            return lev_losses, div_score

        # @tf.function
        # def env_train_step(data, model):

        #     masked_inputs, masked_outputs, preds, total_rewards = data

        #     with tf.GradientTape() as tape1:

        #         # 예측
        #         enc_outputs, _, dec_outputs, _ = model((masked_inputs, masked_outputs), training = True)

        #         # 손실 및 정확도 계산
        #         losses = env_loss_function(preds, dec_outputs)

        #         # 최종 손실
        #         total_losses = losses

        #         # # 최종 손실
        #         # total_losses = tf.matmul(total_rewards[tf.newaxis, :], losses[:, tf.newaxis])

        #     # 최적화
        #     gradients = tape1.gradient(total_losses, model.trainable_variables)
        #     gradients = [tf.clip_by_norm(grads, 1.0) for grads in gradients]
        #     optimizers.apply_gradients(zip(gradients, model.trainable_variables))

        #     return losses

    '''
    AutoEncoding Transformer 손실 함수
    '''
    sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
    def aet_loss_function(real, pred):
        # [PAD] 토큰들 (i.e., 값이 0인 token들)은 무시하는 mask 정의
        mask = tf.math.logical_not( tf.cast( tf.cast(tf.math.equal(real, 0), dtype = tf.int32), dtype = tf.bool ) )
        # # [MASK] 토큰들 (i.e., 값이 4인 token들)은 무시하는 mask 정의
        # mask = tf.math.logical_not( tf.cast( tf.cast(tf.math.equal(real, 0), dtype = tf.int32) + tf.cast(tf.math.equal(real, 4), dtype = tf.int32) , dtype = tf.bool ) )
        losses = sparse_categorical_cross_entropy(real, pred)                         # SparseCategoricalCrossentropy를 활용하여 loss함수 정의

        mask = tf.cast(mask, dtype = losses.dtype)
        losses *= mask

        sum_losses = tf.reduce_sum(losses, axis = 1)
        sum_mask = tf.reduce_sum(mask, axis = 1)

        return tf.reduce_mean(losses), tf.reduce_mean(sum_losses/sum_mask)

    '''
    AutoEncoding Transformer 정확도 함수
    '''
    def aet_accuracy_function(real, pred):
        real = tf.cast(real, dtype = tf.int32)

        # 예측 토큰 반환
        max_pred = tf.argmax(pred, axis = -1)
        max_pred = tf.cast(tf.squeeze(max_pred), dtype = tf.int32)

        # 맞춘 토큰 행렬 (hit_matrix) 구축
        hit_index_mat = tf.cast(tf.where(real == max_pred), dtype = tf.int32)
        if len(hit_index_mat) == 0:
            num_hits = 0
        else:
            # hit_matrix = tf.scatter_nd(hit_index_mat, np.repeat(1, hit_index_mat.shape[0]), shape = real.shape)
            hit_matrix = tf.scatter_nd(hit_index_mat, tf.repeat(1, tf.shape(hit_index_mat)[0]), shape = tf.shape(real))
            num_hits = tf.reduce_sum(hit_matrix, axis = -1)            

        # padding 토큰 (token 0)에 대해서 masking된 행렬 구축
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        num_targets_without_padding = tf.reduce_sum(tf.cast(mask, dtype = tf.int32), axis = -1)

        # 각 sequence 별로 padding 제외 토큰들 중에서 맞춘 비율 계산
        acc = num_hits / num_targets_without_padding
        mean_acc = tf.reduce_mean(acc)
        return tf.cast(mean_acc, dtype = tf.float32)

    '''
    RL-Agent 손실 함수
    '''
    sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
    def lev_loss_function(real, pred):
        losses = sparse_categorical_cross_entropy(real, pred)
        return tf.reduce_mean(losses, axis = -1)

    '''
    Env-Gen 손실 함수
    '''
    sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
    def env_loss_function(real, pred):
        # [PAD] 토큰들 (i.e., 값이 0인 token들)은 무시하는 mask 정의
        mask = tf.math.logical_not( tf.cast( tf.cast(tf.math.equal(real, 0), dtype = tf.int32), dtype = tf.bool ) )
        # # [MASK] 토큰들 (i.e., 값이 4인 token들)은 무시하는 mask 정의
        # mask = tf.math.logical_not( tf.cast( tf.cast(tf.math.equal(real, 0), dtype = tf.int32) + tf.cast(tf.math.equal(real, 4), dtype = tf.int32) , dtype = tf.bool ) )
        losses = sparse_categorical_cross_entropy(real, pred)                         # SparseCategoricalCrossentropy를 활용하여 loss함수 정의

        mask = tf.cast(mask, dtype = losses.dtype)
        losses *= mask

        return tf.reduce_mean(losses, axis = -1)
# ----------------------------------------------------------------------------------------------------------------------------------------------- #
# 모델 훈련
## 1-1. 우도 보상함수 모델 훈련
# if train_for == 'reward_function1':

#     # ---------------------------------------------------------------------------------------- #
#     # 1) 보상함수 훈련
#     # 보상함수 모델 초기화
#     lik_reward_function = Reward_Function(**kwargs)

#     # 메트릭
#     metrics_names = ['re_loss', 're_acc']
#     train_pre_acc = train_pre_loss = 0.0
#     val_pre_acc = max_val_pre_acc = val_pre_loss = 0.0
#     k = 5
#     patience_list = []

#     train_loss_history = []
#     val_loss_history = []
#     train_acc_history = []
#     val_acc_history = []
    
#     # 인풋 시퀀스 마스킹 파라미터인 mean_num_mask (= 평균 마스크 갯수) 정의
#     mean_num_mask = get_masking_param(train_input_sequence)

#     # 훈련 루프
#     for epoch in range(kwargs['num_epochs']):

#         # ---------------------------------------------------------------------------------------- #

#         # 학습 데이터 셋팅
#         real_train_inputs, real_train_labels, fake_train_inputs, fake_train_labels = create_fake_dataset(train_input_sequence, fake_gen_type = 'half-and-half')      # 진짜 / 가짜 시퀀스 및 라벨 생성

#         # 인풋 시퀀스 마스킹
#         '''
#         real_inputs에게 일종의 degree_of_freedom을 부여하기 위해 마스킹을 해주기 --> 추후에 lik_reward_function이 해야할 역할은 "한번도 본적 없는 생성샘플"에 대해서 해당 샘플로부터 "진짜/가짜"를 구분하는 것이다.
#         이 때, lik_reward_function을 학습시키는 단계에서 real_inputs을 단순히 "진짜"라는 라벨로 분류하도록 학습시키면, 이 모델은 그 real_inputs에 overfit 되어버린다. 
#         다시 말해, real_inputs에서 약간 다른 "진짜 같은" 생성샘플에 대해서 모조리 "가짜"라는 라벨로 분류하게 된다는 의미이다. 
#         따라서, 우리는 이러한 overfitting을 예방하고자 real_inputs에 masking을 도입하여, lik_reward_function이 real_inputs과 부분적으로만 다른 "그럴듯한" 샘플들도 "진짜"라고 분류할 수 있도록 유도하고자 하였다.
#         '''
#         mask_lev = mask_level_seletor(thredshold = 0.5)     # token-level masking vs. span-level masking
#         real_train_inputs, masking_idx = poisson_mask_generator(real_train_inputs, lambda_ = mean_num_mask, token_dict = token_dict, masking_level = mask_lev)
#         total_train_inputs = tf.concat([real_train_inputs, fake_train_inputs], axis = 0)                           # 전체 시퀀스 (real = 1, fake = 0)
#         total_train_labels = tf.concat([real_train_labels, fake_train_labels], axis = 0)                                                      # 전체 라벨

#         # 검증 데이터 셋팅
#         real_val_inputs, real_val_labels, fake_val_inputs, fake_val_labels = create_fake_dataset(val_input_sequence, fake_gen_type = 'half-and-half')      # 진짜 / 가짜 시퀀스 및 라벨 생성
#         total_val_inputs = tf.concat([real_val_inputs, fake_val_inputs], axis = 0)                           # 전체 시퀀스 (real = 1, fake = 0)
#         total_val_labels = tf.concat([real_val_labels, fake_val_labels], axis = 0)                                                      # 전체 라벨

#         # Dataset 객체 자체는 cpu에 할당
#         with tf.device("/cpu:0"):
#             train_dataset = tf.data.Dataset.from_tensor_slices((total_train_inputs, total_train_labels)).shuffle(buffer_size = total_train_inputs.shape[0], reshuffle_each_iteration = False)
#             train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
#             train_batchset = train_batchset.prefetch(1)

#             val_dataset = tf.data.Dataset.from_tensor_slices((total_val_inputs, total_val_labels)).shuffle(buffer_size = total_val_inputs.shape[0], reshuffle_each_iteration = False)
#             val_batchset = val_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
#             val_batchset = val_batchset.prefetch(1)

#         # ---------------------------------------------------------------------------------------- #

#         train_cumul_loss = 0        
#         num_val_batch = len(list(val_batchset))
#         val_cumul_loss = 0

#         print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epochs']))
#         pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)

#         # 훈련 배치 루프
#         for idx, (total_train_inputs, total_train_labels) in enumerate(train_batchset):        
#             train_loss, train_acc = train_step((total_train_inputs, total_train_labels), lik_reward_function)

#             # 메트릭 값 업데이트
#             metric_values = [('re_loss', train_loss), ('re_acc', train_acc)]
#             pb_i.update(idx+1, values = metric_values)

#             # 배치별 정확도 누계
#             train_cumul_loss += train_loss.numpy()

#         # 전체 평균 정확도 (훈련셋)
#         train_mean_loss = train_cumul_loss/(idx + 1)
#         train_mean_acc = train_acc.numpy()

#         # 훈련 성능 출력
#         train_acc_delta = train_mean_acc - train_pre_acc
#         print('train_mean_loss : {}, train_mean_acc : {}, train_pre_acc : {}, train_acc_delta : {}'.format(train_mean_loss, train_mean_acc, train_pre_acc, train_acc_delta))
#         train_pre_acc = train_mean_acc

#         # 매 에폭마다 정확도 지표 리셋
#         train_acc_metric.reset_state()

#         # 검증 배치 루프
#         for idx, (total_val_inputs, total_val_labels) in enumerate(val_batchset):        
#             val_loss, val_acc = test_step((total_val_inputs, total_val_labels), lik_reward_function)

#             # 배치별 정확도 누계
#             val_cumul_loss += val_loss.numpy()

#         # 전체 평균 정확도 (검증셋)
#         val_mean_loss = val_cumul_loss/(idx + 1)
#         val_mean_acc = val_acc.numpy()

#         # 매 에폭마다 정확도 지표 리셋
#         val_acc_metric.reset_state()

#         # 검증 성능 출력
#         val_acc_delta = val_mean_acc - val_pre_acc
#         print('val_mean_loss : {}, val_mean_acc : {}, val_pre_acc : {}, val_acc_delta : {}, max_val_pre_acc : {}'.format(val_mean_loss, val_mean_acc, val_pre_acc, val_acc_delta, max_val_pre_acc))
#         val_pre_acc = val_mean_acc

#         # 가중치 저장 조건
#         '''
#         validation_set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
#         '''
#         max_val_acc_delta = val_mean_acc - max_val_pre_acc
#         if max_val_acc_delta > 0.0:

#             # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 이전 정확도 값 업데이트
#             max_val_pre_acc = val_mean_acc

#             # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
#             save_dir = save_weight_dir + '/' + model_name + epoch_path
#             createFolder(save_dir)
#             lik_reward_function.save_weights(save_dir + '/weights.ckpt')

#         # 훈련 / 검증 셋 손실 히스토리 저장
#         train_loss_history += [train_mean_loss]
#         val_loss_history += [val_mean_loss]
#         loss_history_pd = pd.DataFrame(zip(train_loss_history, val_loss_history), columns = ['train_loss', 'val_loss'])
#         loss_history_pd.to_csv(save_result_dir + '/reward_function/lik-loss_history.csv', index_label = 'epoch')

#         # 훈련 / 검증 셋 정확도 히스토리 저장
#         train_acc_history += [train_mean_acc]
#         val_acc_history += [val_mean_acc]
#         acc_history_pd = pd.DataFrame(zip(train_acc_history, val_acc_history), columns = ['train_acc', 'val_acc'])
#         acc_history_pd.to_csv(save_result_dir + '/reward_function/lik-acc_history.csv', index_label = 'epoch')

#         # 학습 중단 조건
#         '''
#         validation_set에 대해서 이전 k-epoch 동안 성능이 연속으로 저하되거나 훈련/검증 정확도 지표가 모두 0.999를 넘을 경우 경우 중단
#         '''
#         if len(patience_list) < k:
#             patience_list += [val_acc_delta]
#         else:
#             del patience_list[0]
#             patience_list += [val_acc_delta]            
#         print('patience_list :', patience_list)
#         if len(np.where(np.array(patience_list) < 0)[0]) == k or (train_mean_acc + val_mean_acc) > (2 * 0.999):
#             break;

# ----------------------------------------------------------------------------------------------------------------------------------------------- #
## 1-1. 속성 보상함수 모델 훈련
if train_for == 'reward_function2':
    # ---------------------------------------------------------------------------------------- #sa
    # 1) 보상함수 훈련
    # 보상함수 모델 초기화
    num_attributes = len(np.unique(train_attribute))
    print('num_attributes : {}'.format(num_attributes))
    attr_reward_function = Reward_Function(num_attributes, **kwargs)

    # 메트릭
    metrics_names = ['re_loss', 're_acc']
    train_pre_acc = train_pre_loss = 0.0
    val_pre_acc = max_val_pre_acc = val_pre_loss = 0.0
    k = 5
    patience_list = []

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # Dataset 객체 자체는 cpu에 할당
    with tf.device("/cpu:0"):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_input_sequence, train_attribute)).shuffle(buffer_size = train_input_sequence.shape[0], reshuffle_each_iteration = False)
        train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
        train_batchset = train_batchset.prefetch(1)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_input_sequence, val_attribute)).shuffle(buffer_size = val_input_sequence.shape[0], reshuffle_each_iteration = False)
        val_batchset = val_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
        val_batchset = val_batchset.prefetch(1)

    # 훈련 루프
    for epoch in range(kwargs['num_epochs']):

        train_cumul_loss = 0        
        num_val_batch = len(list(val_batchset))
        val_cumul_loss = 0

        print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epochs']))
        pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)

        # 훈련 배치 루프
        for idx, (train_inputs, train_labels) in enumerate(train_batchset):        
            train_loss, train_acc = train_step((train_inputs, train_labels), attr_reward_function)

            # 메트릭 값 업데이트
            metric_values = [('re_loss', train_loss), ('re_acc', train_acc)]
            pb_i.update(idx+1, values = metric_values)

            # 배치별 정확도 누계
            train_cumul_loss += train_loss.numpy()

        # 전체 평균 정확도 (훈련셋)
        train_mean_loss = train_cumul_loss/(idx + 1)
        train_mean_acc = train_acc.numpy()

        # 훈련 성능 출력
        train_acc_delta = train_mean_acc - train_pre_acc
        print('train_mean_loss : {}, train_mean_acc : {}, train_pre_acc : {}, train_acc_delta : {}'.format(train_mean_loss, train_mean_acc, train_pre_acc, train_acc_delta))
        train_pre_acc = train_mean_acc

        # 매 에폭마다 정확도 지표 리셋
        train_acc_metric.reset_state()

        # 검증 배치 루프
        for idx, (val_inputs, val_labels) in enumerate(val_batchset):        
            val_loss, val_acc = test_step((val_inputs, val_labels), attr_reward_function)

            # 배치별 정확도 누계
            val_cumul_loss += val_loss.numpy()

        # 전체 평균 정확도 (검증셋)
        val_mean_loss = val_cumul_loss/(idx + 1)
        val_mean_acc = val_acc.numpy()

        # 매 에폭마다 정확도 지표 리셋
        val_acc_metric.reset_state()

        # 검증 성능 출력
        val_acc_delta = val_mean_acc - val_pre_acc
        print('val_mean_loss : {}, val_mean_acc : {}, val_pre_acc : {}, val_acc_delta : {}, max_val_pre_acc : {}'.format(val_mean_loss, val_mean_acc, val_pre_acc, val_acc_delta, max_val_pre_acc))
        val_pre_acc = val_mean_acc

        # 가중치 저장 조건
        '''
        validation_set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
        '''
        max_val_acc_delta = val_mean_acc - max_val_pre_acc
        if max_val_acc_delta > 0.0:

            # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 이전 정확도 값 업데이트
            max_val_pre_acc = val_mean_acc

            # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
            save_dir = save_weight_dir + '/' + model_name + task_path + epoch_path + batch_size_path
            createFolder(save_dir)
            attr_reward_function.save_weights(save_dir + '/weights.ckpt')

        # 훈련 / 검증 셋 손실 히스토리 저장
        train_loss_history += [train_mean_loss]
        val_loss_history += [val_mean_loss]
        loss_history_pd = pd.DataFrame(zip(train_loss_history, val_loss_history), columns = ['train_loss', 'val_loss'])
        loss_history_pd.to_csv(save_result_dir + '/reward_function/' + str(target_task) + '_attr-loss_history.csv', index_label = 'epoch')

        # 훈련 / 검증 셋 정확도 히스토리 저장
        train_acc_history += [train_mean_acc]
        val_acc_history += [val_mean_acc]
        acc_history_pd = pd.DataFrame(zip(train_acc_history, val_acc_history), columns = ['train_acc', 'val_acc'])
        acc_history_pd.to_csv(save_result_dir + '/reward_function/' + str(target_task) +'_attr-acc_history.csv', index_label = 'epoch')

        # 학습 중단 조건
        '''
        validation_set에 대해서 이전 k-epoch 동안 성능이 연속으로 저하되거나 훈련/검증 정확도 지표가 모두 0.999를 넘을 경우 경우 중단
        '''
        if len(patience_list) < k:
            patience_list += [val_acc_delta]
        else:
            del patience_list[0]
            patience_list += [val_acc_delta]            
        print('patience_list :', patience_list)
        if len(np.where(np.array(patience_list) < 0)[0]) == k or (train_mean_acc + val_mean_acc) > (2 * 0.999):
            break;

# ----------------------------------------------------------------------------------------------------------------------------------------------- #
elif train_for == 'Env_Model':
    '''
    'M'asked and 'A'uto-regressive with 'P'rompt Language Model
    인코더에 Masking, 디코더에서 Prompting을 사용하는 Efficient Controllable Transformer 훈련
    '''

    # 오토인코딩 트랜스포머 정의
    ae_transformer = AETransformer(**kwargs)

    # 보상 클래스 토큰 준비
    train_reward_class_vector = np.ones(shape = train_attribute.shape).astype(np.int32)
    for reward_class in range(len(np.unique(train_attribute))):
        train_reward_class_vector[np.where(train_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(reward_class) + ']')

    val_reward_class_vector = np.ones(shape = val_attribute.shape).astype(np.int32)
    for reward_class in range(len(np.unique(val_attribute))):
        val_reward_class_vector[np.where(val_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(reward_class) + ']')

    # 프롬프트 벡터 (= 보상 클래스 토큰 + [BOS] 토큰) 준비
    train_attr_tokens = train_reward_class_vector[:, np.newaxis]
    val_attr_tokens = val_reward_class_vector[:, np.newaxis]

    # 훈련 메트릭
    metrics_names = [str(model_name) + '_loss', str(model_name) + '_acc']
    train_pre_acc = train_pre_loss = 0.0
    val_pre_acc = max_val_pre_acc = val_pre_loss = 0.0 
    k = 5
    patience_list = []
    acc_delta = 1.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    total_start_time = time.time()
    # 훈련 루프
    for epoch in range(kwargs['num_epochs']):
        start_time = time.time()

        # 인풋 시퀀스 마스킹
        mask_lev = mask_level_seletor(thredshold = 0.5)     # token-level masking vs. span-level masking

        if target_task == 'ST':
            '''
            ST의 경우, train_input_sequence와 train_output_sequence가 모두 masking 되므로, masked_train_input_sequence = train_inputs = train_outputs 임.
            '''
            masked_train_input_sequence, masking_idx = poisson_mask_generator(train_input_sequence, lambda_ = mean_num_mask, token_dict = token_dict, masking_level = mask_lev)
            train_inputs = copy.deepcopy(masked_train_input_sequence)

            masked_val_input_sequence, masking_idx = poisson_mask_generator(val_input_sequence, lambda_ = mean_num_mask, token_dict = token_dict, masking_level = mask_lev)
            val_inputs = copy.deepcopy(masked_val_input_sequence)

            if model_name == 'BART':
                train_outputs = copy.deepcopy(train_inputs[:, :train_inputs.shape[1]-1])
                train_outputs = np.concatenate([train_attr_tokens, train_outputs], axis = -1)
                train_targets = copy.deepcopy(train_input_sequence[:, 1:train_input_sequence.shape[1]])
                train_targets = np.concatenate([train_attr_tokens, train_targets], axis = -1)

                val_outputs = copy.deepcopy(val_inputs[:, :val_inputs.shape[1]-1])
                val_outputs = np.concatenate([val_attr_tokens, val_outputs], axis = -1)
                val_targets = copy.deepcopy(val_input_sequence[:, 1:val_input_sequence.shape[1]])
                val_targets = np.concatenate([val_attr_tokens, val_targets], axis = -1)

            elif model_name == 'NAR':
                train_outputs = np.concatenate([train_attr_tokens, train_inputs], axis = -1)
                train_targets = copy.deepcopy(train_input_sequence)
                train_targets = np.concatenate([train_attr_tokens, train_targets], axis = -1)

                val_outputs = np.concatenate([val_attr_tokens, val_inputs], axis = -1)
                val_targets = copy.deepcopy(val_input_sequence)
                val_targets = np.concatenate([val_attr_tokens, val_targets], axis = -1)


        elif target_task == 'DR':
            '''
            DR의 경우, train_output_sequence만 masking 되고 train_input_sequence는 그대로 유지
            '''
            masked_train_output_sequence, masking_idx = poisson_mask_generator(train_output_sequence, lambda_ = mean_num_mask, token_dict = token_dict, masking_level = mask_lev)
            train_outputs = copy.deepcopy(masked_train_output_sequence)

            masked_val_output_sequence, masking_idx = poisson_mask_generator(val_output_sequence, lambda_ = mean_num_mask, token_dict = token_dict, masking_level = mask_lev)
            val_outputs = copy.deepcopy(masked_val_output_sequence)

            if model_name == 'BART':
                train_inputs = copy.deepcopy(train_input_sequence)
                train_outputs = np.concatenate([train_attr_tokens, train_outputs[:, :train_outputs.shape[1]-1]], axis = -1)
                train_targets = copy.deepcopy(train_output_sequence[:, 1:train_output_sequence.shape[1]])
                train_targets = np.concatenate([train_attr_tokens, train_targets], axis = -1)

                val_inputs = copy.deepcopy(val_input_sequence)
                val_outputs = np.concatenate([val_attr_tokens, val_outputs[:, :val_outputs.shape[1]-1]], axis = -1)
                val_targets = copy.deepcopy(val_output_sequence[:, 1:val_output_sequence.shape[1]])
                val_targets = np.concatenate([val_attr_tokens, val_targets], axis = -1)

            elif model_name == 'NAR':
                train_inputs = copy.deepcopy(train_input_sequence)
                train_outputs = np.concatenate([train_attr_tokens, train_outputs], axis = -1)
                train_targets = copy.deepcopy(train_output_sequence)
                train_targets = np.concatenate([train_attr_tokens, train_targets], axis = -1)

                val_inputs = copy.deepcopy(val_input_sequence)
                val_outputs = np.concatenate([val_attr_tokens, val_outputs], axis = -1)
                val_targets = copy.deepcopy(val_output_sequence)
                val_targets = np.concatenate([val_attr_tokens, val_targets], axis = -1)

            

        # Dataset 객체 자체는 cpu에 할당
        with tf.device("/cpu:0"):
            train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs, train_targets)).shuffle(buffer_size = train_inputs.shape[0], reshuffle_each_iteration = False)
            train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
            train_batchset = train_batchset.prefetch(1)

            val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_outputs, val_targets)).shuffle(buffer_size = val_inputs.shape[0], reshuffle_each_iteration = False)
            val_batchset = val_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
            val_batchset = val_batchset.prefetch(1)


        print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epochs']))
        pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)

        # 훈련
        # Train decoder only
        train_cumul_acc = val_cumul_acc = 0
        train_cumul_loss = val_cumul_loss = 0

        # 훈련 배치 루프
        for idx, (train_inputs, train_outputs, train_targets) in enumerate(train_batchset):        
            
            train_loss, train_acc = train_step((train_inputs, train_outputs, train_targets), ae_transformer)

            # 배치별 정확도 및 손실 누계
            train_cumul_acc += train_acc.numpy()
            train_cumul_loss += train_loss.numpy()

            # 메트릭 값 업데이트
            metric_values = [(str(model_name) + '_loss', train_loss), (str(model_name) + '_acc', train_acc)]
            pb_i.update(idx+1, values = metric_values)


        # 전체 평균 정확도 및 손실 (훈련셋)
        train_mean_acc = train_cumul_acc/(idx + 1)
        train_mean_loss = train_cumul_loss/(idx + 1)

        # 훈련 성능 출력
        train_acc_delta = train_mean_acc - train_pre_acc
        print('train_mean_loss : {}, train_mean_acc : {}, train_pre_acc : {}, train_acc_delta : {}'.format(train_mean_loss, train_mean_acc, train_pre_acc, train_acc_delta))
        train_pre_acc = train_mean_acc

        # 검증 배치 루프
        for idx, (total_val_inputs, total_val_outputs, total_val_targets) in enumerate(val_batchset):   
            val_loss, val_acc = test_step((total_val_inputs, total_val_outputs, total_val_targets), ae_transformer)

            # 배치별 정확도 및 손실 누계
            val_cumul_acc += val_acc.numpy()
            val_cumul_loss += val_loss.numpy()

        # 전체 평균 정확도 및 손실 (검증셋)
        val_mean_loss = val_cumul_loss/(idx + 1)
        val_mean_acc = val_cumul_acc/(idx + 1)

        # 검증 성능 출력
        val_acc_delta = val_mean_acc - val_pre_acc
        print('val_mean_loss : {}, val_mean_acc : {}, val_pre_acc : {}, val_acc_delta : {}, max_val_pre_acc : {}'.format(val_mean_loss, val_mean_acc, val_pre_acc, val_acc_delta, max_val_pre_acc))
        val_pre_acc = val_mean_acc

        # 가중치 저장 조건
        '''
        validation_set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
        '''
        max_val_acc_delta = val_mean_acc - max_val_pre_acc
        if max_val_acc_delta > 0.0:

            # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 이전 정확도 값 업데이트
            max_val_pre_acc = val_mean_acc

            # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
            save_dir = save_weight_dir + '/' + model_name + task_path + epoch_path + batch_size_path
            createFolder(save_dir)
            ae_transformer.save_weights(save_dir + '/weights.ckpt')

        # 훈련 / 검증 셋 손실 히스토리 저장
        train_loss_history += [train_mean_loss]
        val_loss_history += [val_mean_loss]
        loss_history_pd = pd.DataFrame(zip(train_loss_history, val_loss_history), columns = ['train_loss', 'val_loss'])
        file_dir = save_result_dir + '/' + str(model_name)
        file_name = '/loss_history' + str(task_path) + '.csv'
        loss_history_pd.to_csv(file_dir + file_name, index_label = 'epoch')

        # 훈련 / 검증 셋 정확도 히스토리 저장
        train_acc_history += [train_mean_acc]
        val_acc_history += [val_mean_acc]
        acc_history_pd = pd.DataFrame(zip(train_acc_history, val_acc_history), columns = ['train_acc', 'val_acc'])
        file_dir = save_result_dir + '/' + str(model_name)
        file_name = '/acc_history' + str(task_path) + '.csv'
        acc_history_pd.to_csv(file_dir + file_name, index_label = 'epoch')

        # 학습 중단 조건
        '''
        validation_set에 대해서 이전 k-epoch 동안 성능이 연속으로 저하되거나 훈련/검증 정확도 지표가 모두 0.999를 넘을 경우 경우 중단
        '''
        if len(patience_list) < k:
            patience_list += [val_acc_delta]
        else:
            del patience_list[0]
            patience_list += [val_acc_delta]            
        print('patience_list :', patience_list)
        if len(np.where(np.array(patience_list) < 0)[0]) == k or (train_mean_acc + val_mean_acc) > (2 * 0.999):
            break;

        end_time = time.time()
        cur_sec = (end_time - start_time)%60
        cur_min = ((end_time - start_time)//60)%60
        cur_hr = ((end_time - start_time)//60)//60
        total_sec = (end_time - total_start_time)%60
        total_min = ((end_time - total_start_time)//60)%60
        total_hr = ((end_time - total_start_time)//60)//60
        print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
        print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))


# 5. RL 에이전트 훈련
elif train_for == 'RL_Agent':

    # Dataset 객체 자체는 cpu에 할당
    if target_task == 'ST':

        # num_attribute 사전 선언
        num_attributes = len(np.unique(train_attribute))

        # "리버스" 보상 클래스 토큰 벡터 준비
        train_reward_class_vector = np.ones(shape = train_attribute.shape).astype(np.int32)
        for reward_class in range(len(np.unique(train_attribute))):
            rev_reward_class = np.unique(train_attribute)[np.where(reward_class != np.unique(train_attribute))[0]][0]
            train_reward_class_vector[np.where(train_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(rev_reward_class) + ']')

        # 프롬프트 벡터 (= 보상 클래스 토큰 + [BOS] 토큰) 준비
        train_attr_tokens = train_reward_class_vector[:, np.newaxis]


        with tf.device("/cpu:0"):
            '''
            데이터셋 구축
            '''
            dataset = tf.data.Dataset.from_tensor_slices((train_input_sequence, train_attr_tokens, train_attribute)).shuffle(buffer_size = train_input_sequence.shape[0], reshuffle_each_iteration = False)
            batchset = dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
            batchset = batchset.prefetch(1)

    elif target_task == 'DR':

        # num_attribute 사전 선언
        num_attributes = len(np.unique(train_attribute))

        # sample x drug 데이터 로드
        train_dat = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/prep_data/drug-discovery/train_dat.csv', index_col = 0)

        # reposition_code_mapping
        reposition_mapper = np.array([[2, 4], [0, 5], [3, 5], [3, 1]])

        # reposition이 존재하는 경우 (= code: 2, 0, 3)만 솎아내기
        total_target_idx = []
        for idx, original_attribute in enumerate(reposition_mapper[:, 0]):
            target_idx = list(np.where(train_attribute == original_attribute)[0])
            total_target_idx += target_idx
        train_input_sequence = train_input_sequence[np.array(total_target_idx)]
        train_output_sequence = train_output_sequence[np.array(total_target_idx)]
        train_attribute = train_attribute[np.array(total_target_idx)]
        train_dat = train_dat.iloc(axis =0)[np.array(total_target_idx)]

        # 보상 클래스 토큰 벡터 준비
        # control_attribute 만들어주기 (2 -> 4, 0 -> 5, 3_1 -> 5, 3_2 -> 1)
        control_attribute = copy.deepcopy(train_attribute)
        for idx, original_attribute in enumerate(reposition_mapper[:, 0]):
            control_attribute[np.where(train_attribute == original_attribute)[0]] = reposition_mapper[idx, 1]
        control_attribute[np.where(train_dat['Related drug'] == 'Topiramate')[0]] = int(5)
        control_attribute[np.where(train_dat['Related drug'] == 'Valproic acid')[0]] = int(1)
        control_attribute = tf.cast(control_attribute, dtype = tf.int32)

        # 프롬프트 벡터 (= 보상 클래스 토큰 + [BOS] 토큰) 준비
        train_reward_class_vector = []
        for code in np.unique(control_attribute):
            attribute_label = '[R_' + str(code) + ']'
            attribute_token = get_token(token_dict, attribute_label)
            train_reward_class_vector += [attribute_token] * len(np.where(control_attribute == code)[0])
             
        train_attr_tokens = np.array(train_reward_class_vector)[:, np.newaxis]

        with tf.device("/cpu:0"):

            '''
            데이터셋 구축 : DR의 경우 train_input_sequnece와 train_output_sequence를 모두 제공
            '''
            dataset = tf.data.Dataset.from_tensor_slices((train_input_sequence, train_output_sequence, train_attr_tokens, control_attribute)).shuffle(buffer_size = train_input_sequence.shape[0], reshuffle_each_iteration = False)
            batchset = dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
            batchset = batchset.prefetch(1)

    # # 표본 샘플링
    # batch_size = kwargs['batch_size']
    # num_batch = 1
    # rand_idx = tf.random.uniform(shape = (batch_size * num_batch, ), minval = 0, maxval = train_input_sequence.shape[0], dtype = tf.int32).numpy()
    # sample_train_input_sequence = train_input_sequence[rand_idx, :]
    # sample_train_attribute = train_attribute[rand_idx]

    # # "리버스" 보상 클래스 토큰 벡터 준비
    # sample_train_reward_class_vector = np.ones(shape = sample_train_attribute.shape).astype(np.int32)
    # for reward_class in range(len(np.unique(sample_train_attribute))):
    #     rev_reward_class = np.unique(sample_train_attribute)[np.where(reward_class != np.unique(sample_train_attribute))[0]][0]
    #     sample_train_reward_class_vector[np.where(sample_train_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(rev_reward_class) + ']')

    # val_reward_class_vector = np.ones(shape = val_attribute.shape).astype(np.int32)
    # for reward_class in range(len(np.unique(val_attribute))):
    #     rev_reward_class = np.unique(val_attribute)[np.where(reward_class != np.unique(val_attribute))[0]][0]
    #     val_reward_class_vector[np.where(val_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(rev_reward_class) + ']')

    # # 프롬프트 벡터 (= 보상 클래스 토큰 + [BOS] 토큰) 준비
    # sample_train_attr_tokens = sample_train_reward_class_vector[:, np.newaxis]

    # # Dataset 객체 자체는 cpu에 할당
    # with tf.device("/cpu:0"):
    #     dataset = tf.data.Dataset.from_tensor_slices((sample_train_input_sequence, sample_train_attr_tokens, sample_train_attribute)).shuffle(buffer_size = sample_train_input_sequence.shape[0], reshuffle_each_iteration = False)
    #     batchset = dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    #     batchset = batchset.prefetch(1)

    # ---------------------------------------------------------------------------------------- #
    # 1) 보상함수 모델 초기화
    # --> 속성 보상함수 관련 파라미터 로드
    with open(hyper_param_dir + '/Reward_Function/kwargs_attr_reward_function' + task_path + epoch_path + batch_size_path, 'r') as f:
        re_kwargs = json.load(f)
    # with open('/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/Reward_Function/kwargs_attr_reward_function_mlp_500', 'r') as f:
    #     re_kwargs = json.load(f)
    attr_reward_function = Reward_Function(num_attributes, **re_kwargs)

    # --> 사전학습된 속성 보상함수 로드
    load_dir1 = save_weight_dir + '/attr_reward_function' + task_path + epoch_path + batch_size_path
    # load_dir1 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/attr_reward_function_mlp_500'
    print(load_dir1)
    # load_dir1 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/attr_reward_function_mlp_500'
    # print(load_dir1)
    attr_reward_function.load_weights(tf.train.latest_checkpoint(load_dir1))

    # ---------------------------------------------------------------------------------------- #
    # 2) Env_Gen 모델 초기화
    # env_gen_model_name = input('decoder (BART vs. NAR) : ')
    env_gen_model_name = 'NAR'
    prefix = hyper_param_dir + '/' + env_gen_model_name
    env_gen_kwargs = '/kwargs_' + env_gen_model_name + task_path + epoch_path + batch_size_path
    env_gen_kwargs_dir = prefix + env_gen_kwargs
    with open(env_gen_kwargs_dir, 'r') as f:
        env_kwargs = json.load(f)

    env_gen_model = AETransformer(**env_kwargs)

    # --> 초기 정책모델 학습 가중치 로드하여 타겟 정책모델 가중치 초기화
    prefix = save_weight_dir + '/'
    env_gen_weights_dir = prefix + env_gen_model_name + task_path + epoch_path + batch_size_path
    print(env_gen_weights_dir)
    # env_gen_weights_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/NAR_ST_500'
    # print(env_gen_weights_dir)
    env_gen_model.load_weights(tf.train.latest_checkpoint(env_gen_weights_dir))
    
    # ---------------------------------------------------------------------------------------- #
    # 3) RL 에이전트 모델 초기화
    lev_agent = LEVAgent(**kwargs)

    # ---------------------------------------------------------------------------------------- #
    # 4) 그 외
    KL_divergence = tf.keras.losses.KLDivergence(reduction = tf.keras.losses.Reduction.NONE)

    # ---------------------------------------------------------------------------------------- #
    # 5) 훈련 루프
    # --> 메트릭 초기화
    metrics_names = ['lev_loss', 'div_score', 'reward', 'cont_rewards', 'attr_rewards']
    pre_reward = max_pre_reward = 0
    reward_history = []
    freq_history = []

    k = 5
    patience_list = []
    acc_delta = 1.0

    # --> 에이전트 모델 외 다른 모델 동결    
    # env_gen_model.trainable = False
    attr_reward_function.trainable = False

    # --> 에이전트 훈련 알고리즘
    algo = kwargs['algo']

    # --> 루프 수행
    total_start_time = time.time()
    for epoch in range(kwargs['num_epochs']):
        print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epochs']))
        pb_i = Progbar(len(batchset), stateful_metrics = metrics_names)

        start_time = time.time()
        cumul_reward = 0
        cumul_cont_reward = 0
        cumul_attr_reward = 0
        cumul_action_token_freqs = 0
        print('reward type is : {}'.format(kwargs['reward']))

        # Fixed eta
        if kwargs['eta'] != 9999:
            eta = kwargs['eta']

        # Scheduling eta
        else:
            '''
            decay_step : # of steps to decay
            m_mul : decay size
            min_eta : minimum value of eta
            max_eta : maximum value of eta
            '''
            if epoch == 0:
                initial_eta = 0.01
                eta, decayed_max_eta = CosineDecayRestart(epoch, max_eta = initial_eta, decay_step = 15, m_mul = 0.9, min_eta = 0.0001)         # decayed_max_eta = initial_eta at epoch 0
            else:
                eta, decayed_max_eta = CosineDecayRestart(epoch, max_eta = decayed_max_eta, decay_step = 15, m_mul = 0.9, min_eta = 0.0001)
            # eta = CosineDecayRestart(epoch, decay_step = 25, m_mul = 0.9, min_eta = 0.0001, max_eta = 0.005)

        if target_task == 'ST':

            for idx, (inputs, attr_tokens, attrs) in enumerate(batchset):        

                # Train encoder & decoder
                enc_pad_mask, _, _ = env_gen_model.mask_generator(inputs, inputs)

                '''
                편집 계획 (edit_plans) 생성
                '''
                if algo == 'PG':
                    agent_outputs, action_att_weights = lev_agent(inputs, enc_pad_mask, training = False)
                    agent_actions, action_tokens = return_action_token(agent_outputs, inputs, token_dict, mode = 'train')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                                # action_tokens = action의 토큰 : (batch_size, seq_len)
                elif algo == 'PPO':
                    if epoch == 0:
                        if idx == 0:
                            old_lev_agent = LEVAgent(**kwargs)
                        agent_outputs, action_att_weights = old_lev_agent(inputs, enc_pad_mask, training = False)
                        agent_actions, action_tokens = return_action_token(agent_outputs, inputs, token_dict, mode = 'train')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                                    # action_tokens = action의 토큰 : (batch_size, seq_len)
                    else:
                        # # 매 epoch의 첫번째 배치마다 old_lev_agent의 가중치를 lev_agent의 가중치로 reset
                        # if idx == 0:
                        # 매 epoch의 5번째 배치마다 old_lev_agent의 가중치를 lev_agent의 가중치로 reset
                        if idx % 40 == 0:
                            old_lev_agent.set_weights(lev_agent.get_weights())

                        # 매 epoch의 두번째 이상의 배치에서는 old_lev_agent의 가중치를 고정
                        agent_outputs, action_att_weights = old_lev_agent(inputs, enc_pad_mask, training = False)
                        agent_actions, action_tokens = return_action_token(agent_outputs, inputs, token_dict, mode = 'train')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                                    # action_tokens = action의 토큰 : (batch_size, seq_len)

                '''
                레반슈타인 연산 수행
                '''
                # 레반슈타인 연산자 적용
                masked_inputs, new_action_tokens = apply_lev_operation(inputs, action_tokens, token_dict)
                masked_outputs = np.concatenate([attr_tokens, masked_inputs], axis = -1)  #   마스크 인풋 시퀀스의 앞에 attr_tokens 붙여주기

                '''
                스타일 전환 : attribute 코드를 prefix하여 환경 모델에 입력
                '''
                # _ ,_, dec_outputs, _ = env_gen_model((masked_inputs, masked_outputs[:, :masked_outputs.shape[1]-1]), training = False)
                _ ,_, dec_outputs, _ = env_gen_model((masked_inputs, masked_outputs), training = False)

                if kwargs['env_sampling'] == 'greedy':
                    gen_seqs = tf.cast(tf.math.argmax(dec_outputs, axis = -1), dtype = tf.int32)
                elif kwargs['env_sampling'] == 'stochastic':
                    long_dec_outputs = tf.reshape(dec_outputs, shape = (-1, tf.shape(dec_outputs)[-1]))
                    gen_seqs = tf.reshape(tf.random.categorical(logits = long_dec_outputs, num_samples = 1), shape = (tf.shape(dec_outputs)[0], tf.shape(dec_outputs)[1]))
                    gen_seqs = tf.cast(gen_seqs, dtype = tf.int32)

                
                '''
                제어된 시퀀스 생성
                '''
                controlled_gens = fill_pad_after_eos(gen_seqs, masked_outputs, token_dict)

                '''
                보상 계산 : 스타일 전환된 시퀀스가 얼마나 그럴듯한지 평가
                '''
                # '[KEP]', '[DEL]', '[REP]' 토큰의 갯수
                kep_ratio = get_num_of_ops(new_action_tokens, inputs, token_dict, '[KEP]')
                # del_ratio = get_num_of_ops(new_action_tokens, inputs, token_dict, '[DEL]')
                # rep_ratio = get_num_of_ops(new_action_tokens, inputs, token_dict, '[REP]')
                # print('kep_ratio : {}, mean_del : {}, mean_rep : {}'.format(tf.reduce_mean(kep_ratio), tf.reduce_mean(del_ratio), tf.reduce_mean(rep_ratio)))

                # # 길이 패널티 (lenth_diff_cost)
                # inputs_len = get_len_of_seq(inputs, token_dict)
                # gens_len = get_len_of_seq(controlled_gens, token_dict)
                # len_diff = gens_len - inputs_len
                # len_buffer = kwargs['len_buffer']
                # over_len_idx = tf.cast(tf.math.greater(len_diff, len_buffer), dtype = tf.float32)
                # over_diff_ratio = tf.cast((len_diff-len_buffer) / gens_len, dtype = tf.float32) * over_len_idx
                # # print('over_diff_ratio :', over_diff_ratio)

                # 내용 보상 (content_reward)
                # postive_value_idx = tf.cast(tf.math.greater(kep_ratio - over_diff_ratio, 0.0), dtype = tf.float32)
                # negative_value_idx = tf.cast(tf.math.less(kep_ratio - over_diff_ratio, 0.0), dtype = tf.float32)
                # cont_rewards = (kep_ratio - over_diff_ratio) * postive_value_idx
                # cont_rewards = kep_ratio - 0.1 * over_diff_ratio
                # cont_rewards = kep_ratio - del_ratio - rep_ratio - over_diff_ratio
                # cont_rewards = kep_ratio
                # print('kep_ratio :', kep_ratio)
                # print('kep_ratio_sum :', tf.reduce_sum(kep_ratio))
                # print('cont_rewards :', cont_rewards)
                # print('positive num : ', tf.math.greater(cont_rewards, 0.0))
                cont_rewards = kep_ratio                   


                # 속성 보상 (attr_reward)
                enc_pad_mask, _, _ = attr_reward_function.mask_generator(controlled_gens, controlled_gens)

                attr_probs = tf.nn.softmax(attr_reward_function(controlled_gens, enc_pad_mask, training = False), axis = -1)
                attrs_onehot = 1 - tf.one_hot(attrs, depth = num_attributes)
                attr_rev_probs = tf.math.multiply(attr_probs, attrs_onehot)

                attr_rewards = tf.reduce_sum(attr_rev_probs, axis = -1)
                attrs_pred = tf.argmax(attr_probs, axis = -1)

                # # 역 번역 보상 (back-translation)
                # _ ,_, dec_outputs, _ = env_gen_model((gen_seqs.numpy()[:, 1:], gen_seqs.numpy()), training = False)
                # bt_rewards = tf.reduce_mean(tf.reduce_max(dec_outputs, axis = -1).numpy()[:, 1:], axis = -1)
                # # print('cont_rewards : {}, attr_rewards : {}, bt_rewards : {}'.format(cont_rewards.shape, attr_rewards.shape, bt_rewards.shape))
                # # print(bt_rewards)

                # 총 보상 계산
                # greater_equal_idx = tf.cast(tf.math.greater_equal(attr_rewards, 0.95), dtype = tf.float32)
                # less_idx = tf.cast(tf.math.less(attr_rewards, 0.95), dtype = tf.float32)
                # total_rewards = tf.math.multiply(cont_rewards + attr_rewards, greater_equal_idx)
                # total_rewards = total_rewards + (-1.0) * less_idx

                if kwargs['reward'] == 'S':
                    # total_rewards = cont_rewards + attr_rewards + bt_rewards                                     # 단순합
                    total_rewards = cont_rewards + attr_rewards                                     # 단순합
                elif kwargs['reward'] == 'A':
                    # total_rewards = (cont_rewards + attr_rewards + bt_rewards)/3                                 # 산술평균
                    total_rewards = (cont_rewards + attr_rewards)/2                                 # 산술평균
                elif kwargs['reward'] == 'G':
                    # total_rewards = tf.math.sqrt(cont_rewards * attr_rewards + bt_rewards)                       # 기하평균
                    total_rewards = tf.math.sqrt(cont_rewards * attr_rewards)                       # 기하평균
                elif kwargs['reward'] == 'H':
                    # total_rewards = (3*cont_rewards*attr_rewards*bt_rewards)/(cont_rewards + attr_rewards + bt_rewards)     # 조화평균
                    total_rewards = (2 * (cont_rewards * attr_rewards))/(cont_rewards + attr_rewards)     # 조화평균
                    print('Harmonic Total Rewards : {}'.format(tf.reduce_mean(total_rewards)))
                
                # Train Edit model
                lev_losses, div_score = agent_train_step((inputs, agent_outputs, agent_actions, total_rewards), lev_agent, enc_pad_mask, eta, algo)

                # # Train Edit & Gen model
                # lev_losses, div_score = agent_train_step((inputs, attr_tokens, agent_actions, total_rewards), lev_agent, enc_pad_mask, token_dict, eta)
                # if epoch >= kwargs['num_epochs']//10:            # 전체 epoch의 1/10 지점부터 Gen model train
                #     env_losses = env_train_step((masked_inputs, masked_outputs, controlled_gens, total_rewards), env_gen_model)


                # 메트릭 값 업데이트
                # print('mean_cont : {}, mean_attr : {}, len_diff : {}'.format(tf.reduce_mean(cont_rewards), tf.reduce_mean(attr_rewards), tf.reduce_mean(len_diff)))
                if tf.math.is_nan(tf.reduce_mean(cont_rewards + attr_rewards)) == True or tf.math.is_nan(tf.reduce_mean(lev_losses)) == True:
                    if tf.math.is_nan(tf.reduce_mean(cont_rewards + attr_rewards)) == True:
                        reward_or_loss = 'cont_rewards + attr_rewards'
                    else:
                        reward_or_loss = 'lev_loss'
                    exit(print('{} is nan !!!!'.format(reward_or_loss)))

                metric_values = [('lev_loss', tf.reduce_mean(lev_losses)), ('div_score', tf.reduce_mean(div_score)), ('reward', tf.reduce_mean(cont_rewards + attr_rewards)), ('cont_rewards', tf.reduce_mean(cont_rewards)), ('attr_rewards', tf.reduce_mean(attr_rewards))]
                pb_i.update(idx+1, values = metric_values)

                # 배치별 보상 누계
                cumul_reward += tf.reduce_mean(cont_rewards + attr_rewards).numpy()
                cumul_cont_reward += tf.reduce_mean(cont_rewards).numpy()
                cumul_attr_reward += tf.reduce_mean(attr_rewards).numpy()

                # 배치별 action_token의 빈도수 통계
                action_token_types, action_token_freqs = get_action_tokens_staistics(action_tokens, token_dict)
                cumul_action_token_freqs += action_token_freqs.numpy()

        elif target_task == 'DR':

            for idx, (inputs, outputs, attr_tokens, attrs) in enumerate(batchset):        

                # Train encoder & decoder
                enc_pad_mask, _, _ = env_gen_model.mask_generator(outputs, outputs)

                '''
                편집 계획 (edit_plans) 생성
                '''
                if algo == 'PG':
                    agent_outputs, action_att_weights = lev_agent(outputs, enc_pad_mask, training = False)
                    agent_actions, action_tokens = return_action_token(agent_outputs, outputs, token_dict, mode = 'train')      # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                                # action_tokens = action의 토큰 : (batch_size, seq_len)
                elif algo == 'PPO':
                    if epoch == 0:
                        if idx == 0:
                            old_lev_agent = LEVAgent(**kwargs)
                        agent_outputs, action_att_weights = old_lev_agent(outputs, enc_pad_mask, training = False)
                        agent_actions, action_tokens = return_action_token(agent_outputs, outputs, token_dict, mode = 'train')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                                    # action_tokens = action의 토큰 : (batch_size, seq_len)
                    else:
                        # # 매 epoch의 첫번째 배치마다 old_lev_agent의 가중치를 lev_agent의 가중치로 reset
                        # if idx == 0:
                        # 매 epoch의 5번째 배치마다 old_lev_agent의 가중치를 lev_agent의 가중치로 reset
                        if idx % 40 == 0:
                            old_lev_agent.set_weights(lev_agent.get_weights())

                        # 매 epoch의 두번째 이상의 배치에서는 old_lev_agent의 가중치를 고정
                        agent_outputs, action_att_weights = old_lev_agent(outputs, enc_pad_mask, training = False)
                        agent_actions, action_tokens = return_action_token(agent_outputs, outputs, token_dict, mode = 'train')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                                    # action_tokens = action의 토큰 : (batch_size, seq_len)

                '''
                레반슈타인 연산 수행
                '''
                # 레반슈타인 연산자 적용
                masked_outputs, new_action_tokens = apply_lev_operation(outputs, action_tokens, token_dict)     # target_task = 'ST' 때와 달리, masked_inputs 없이 masked_outputs만 얻음
                masked_outputs = np.concatenate([attr_tokens, masked_outputs], axis = -1)                       # 마스크 인풋 시퀀스의 앞에 attr_tokens 붙여주기

                '''
                스타일 전환 : attribute 코드를 prefix하여 환경 모델에 입력
                '''
                _ ,_, dec_outputs, _ = env_gen_model((inputs, masked_outputs), training = False)        # target_task = 'ST' 때와 달리, inputs과 masked_outputs을 환경생성 모델의 인풋으로 활용

                if kwargs['env_sampling'] == 'greedy':
                    gen_seqs = tf.cast(tf.math.argmax(dec_outputs, axis = -1), dtype = tf.int32)
                elif kwargs['env_sampling'] == 'stochastic':
                    long_dec_outputs = tf.reshape(dec_outputs, shape = (-1, tf.shape(dec_outputs)[-1]))
                    gen_seqs = tf.reshape(tf.random.categorical(logits = long_dec_outputs, num_samples = 1), shape = (tf.shape(dec_outputs)[0], tf.shape(dec_outputs)[1]))
                    gen_seqs = tf.cast(gen_seqs, dtype = tf.int32)

                '''
                제어된 시퀀스 생성
                '''
                controlled_gens = fill_pad_after_eos(gen_seqs, masked_outputs, token_dict)

                '''
                보상 계산 : 스타일 전환된 시퀀스가 얼마나 그럴듯한지 평가
                '''
                # '[KEP]', '[DEL]', '[REP]' 토큰의 갯수
                kep_ratio = get_num_of_ops(new_action_tokens, outputs, token_dict, '[KEP]')
                cont_rewards = kep_ratio

                # 속성 보상 (attr_reward)
                enc_pad_mask, _, _ = attr_reward_function.mask_generator(controlled_gens, controlled_gens)

                attr_probs = tf.nn.softmax(attr_reward_function(controlled_gens, enc_pad_mask, training = False), axis = -1)
                attrs_onehot = tf.one_hot(attrs, depth = num_attributes)
                attr_probs_hit = tf.math.multiply(attr_probs, attrs_onehot)

                attr_rewards = tf.reduce_sum(attr_probs_hit, axis = -1)
                attrs_pred = tf.argmax(attr_probs, axis = -1)

                # 총 보상 계산
                if kwargs['reward'] == 'S':
                    # total_rewards = cont_rewards + attr_rewards + bt_rewards                                     # 단순합
                    total_rewards = cont_rewards + attr_rewards                                     # 단순합
                elif kwargs['reward'] == 'A':
                    # total_rewards = (cont_rewards + attr_rewards + bt_rewards)/3                                 # 산술평균
                    total_rewards = (cont_rewards + attr_rewards)/2                                 # 산술평균
                elif kwargs['reward'] == 'G':
                    # total_rewards = tf.math.sqrt(cont_rewards * attr_rewards + bt_rewards)                       # 기하평균
                    total_rewards = tf.math.sqrt(cont_rewards * attr_rewards)                       # 기하평균
                elif kwargs['reward'] == 'H':
                    # total_rewards = (3*cont_rewards*attr_rewards*bt_rewards)/(cont_rewards + attr_rewards + bt_rewards)     # 조화평균
                    total_rewards = (2 * (cont_rewards * attr_rewards))/(cont_rewards + attr_rewards)     # 조화평균
                    print('Harmonic Total Rewards : {}'.format(tf.reduce_mean(total_rewards)))
                
                # Train Edit model
                lev_losses, div_score = agent_train_step((outputs, agent_outputs, agent_actions, total_rewards), lev_agent, enc_pad_mask, eta, algo)

                # 메트릭 값 업데이트
                # print('mean_cont : {}, mean_attr : {}, len_diff : {}'.format(tf.reduce_mean(cont_rewards), tf.reduce_mean(attr_rewards), tf.reduce_mean(len_diff)))
                if tf.math.is_nan(tf.reduce_mean(cont_rewards + attr_rewards)) == True or tf.math.is_nan(tf.reduce_mean(lev_losses)) == True:
                    if tf.math.is_nan(tf.reduce_mean(cont_rewards + attr_rewards)) == True:
                        reward_or_loss = 'cont_rewards + attr_rewards'
                    else:
                        reward_or_loss = 'lev_loss'
                    exit(print('{} is nan !!!!'.format(reward_or_loss)))

                metric_values = [('lev_loss', tf.reduce_mean(lev_losses)), ('div_score', tf.reduce_mean(div_score)), ('reward', tf.reduce_mean(cont_rewards + attr_rewards)), ('cont_rewards', tf.reduce_mean(cont_rewards)), ('attr_rewards', tf.reduce_mean(attr_rewards))]
                pb_i.update(idx+1, values = metric_values)

                # 배치별 보상 누계
                cumul_reward += tf.reduce_mean(cont_rewards + attr_rewards).numpy()
                cumul_cont_reward += tf.reduce_mean(cont_rewards).numpy()
                cumul_attr_reward += tf.reduce_mean(attr_rewards).numpy()

                # 배치별 action_token의 빈도수 통계
                action_token_types, action_token_freqs = get_action_tokens_staistics(action_tokens, token_dict)
                cumul_action_token_freqs += action_token_freqs.numpy()

        for my_idx in range(5):
            print('inputs : {}'.format([token_dict[t] for t in inputs[my_idx, :].numpy()]))
            print('edit plans : {}'.format([token_dict[t] for t in action_tokens[my_idx, :].numpy()]))
            print('true_attrs_label : {}'.format(attrs[my_idx]))
            print('gens : {}'.format([token_dict[t] for t in controlled_gens[my_idx, :].numpy()]))
            print('pred_attrs_label : {}'.format(attrs_pred[my_idx]))
            print('\n')

        # 전체 평균 보상
        mean_reward = cumul_reward/(idx + 1)
        mean_cont_reward = cumul_cont_reward/(idx + 1)
        mean_attr_reward = cumul_attr_reward/(idx + 1)

        # 전체 평균 action_token 빈도수 통계
        mean_freq = cumul_action_token_freqs/(idx + 1)

        # 보상 히스토리 저장
        reward_history += [mean_reward]
        reward_history_pd = pd.DataFrame(reward_history, columns = ['reward'])
        file_dir = save_result_dir + '/' + model_name
        file_name = '/reward_history_continue' + task_path + '_epoch=' + str(kwargs['num_epochs']) + '_opt=' + OptSchedule + '_lr=' + str(kwargs['lr']) + '_lb=' + str(kwargs['len_buffer']) + '_eta=' + str(kwargs['eta']) + '_es=' + kwargs['env_sampling'] + '_reward=' + kwargs['reward'] + '_algo=' + kwargs['algo'] + '_early_stop=' + kwargs['early_stop'] + '.csv'
        reward_history_pd.to_csv(file_dir + file_name, index_label = 'epoch')

        # action_token 빈도수 통계 히스토리 저장
        freq_history += [mean_freq]
        column_vector = [token_dict[t] for t in action_token_types.numpy()]
        freq_history_pd = pd.DataFrame(freq_history, columns = column_vector)
        file_dir = save_result_dir + '/' + model_name
        file_name = '/freq_history_continue' + task_path + '_epoch=' + str(kwargs['num_epochs']) + '_opt=' + OptSchedule + '_lr=' + str(kwargs['lr']) + '_lb=' + str(kwargs['len_buffer']) + '_eta=' + str(kwargs['eta']) + '_es=' + kwargs['env_sampling'] + '_reward=' + kwargs['reward'] + '_algo=' + kwargs['algo'] + '_early_stop=' + kwargs['early_stop'] + '.csv'
        freq_history_pd.to_csv(file_dir + file_name, index_label = 'epoch')

        end_time = time.time()
        cur_sec = (end_time - start_time)%60
        cur_min = ((end_time - start_time)//60)%60
        cur_hr = ((end_time - start_time)//60)//60
        total_sec = (end_time - total_start_time)%60
        total_min = ((end_time - total_start_time)//60)%60
        total_hr = ((end_time - total_start_time)//60)//60
        print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
        print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))

        # 가중치 저장 조건
        '''
        이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
        '''
        max_reward_delta = mean_reward - max_pre_reward
        if max_reward_delta > 0.0:

            # 현 보상이 가장 높았던 이전 보상보다 개선됐을 경우에만 이전 보상값 업데이트
            max_pre_reward = mean_reward

            # 현 보상이 가장 높았던 이전 보상보다 개선됐을 경우에만 가중치 저장
            save_dir = save_weight_dir + '/' + model_name + task_path + '_epoch=' + str(kwargs['num_epochs']) + '_opt=' + OptSchedule + '_lr=' + str(kwargs['lr']) + '_lb=' + str(kwargs['len_buffer']) + '_eta=' + str(kwargs['eta']) + '_es=' + kwargs['env_sampling'] + '_reward=' + kwargs['reward'] + '_algo=' + kwargs['algo'] + '_early_stop=' + kwargs['early_stop']
            createFolder(save_dir)
            lev_agent.save_weights(save_dir + '/weights.ckpt')

        # early_stop 인자가  'yes'인 경우,
        if kwargs['early_stop'] == 'yes':
            # 학습 중단 조건
            '''
            이전 k-epoch 동안 보상이 연속으로 저하될 경우 중단
            '''
            reward_delta = mean_reward - pre_reward
            print('mean_reward : {}, mean_cont_reward : {}, mean_attr_reward : {}, pre_reward : {}, reward_delta : {}'.format(mean_reward, mean_cont_reward, mean_attr_reward, pre_reward, reward_delta))
            pre_reward = mean_reward
            if len(patience_list) < k:
                patience_list += [reward_delta]
            else:
                del patience_list[0]
                patience_list += [reward_delta]            
            print('patience_list :', patience_list)

            # 학습 중단
            if len(np.where(np.array(patience_list) < 0)[0]) == k:
                break;

        print('mean_reward : {}, mean_cont_reward : {}, mean_attr_reward : {}, pre_reward : {}'.format(mean_reward, mean_cont_reward, mean_attr_reward, pre_reward))

# %%
