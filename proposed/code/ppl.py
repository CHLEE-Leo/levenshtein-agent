# %%
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Progbar
import time
from utils import *
import pickle, json
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)

parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--my_seed', type = int, required = True)
parser.add_argument('--batch_size', type = int, required = True)
parser.add_argument('--num_epochs', type = int, required = True)
parser.add_argument('--lr', type = float, required = True)
args = parser.parse_args()
kwargs = {
    'batch_size' : args.batch_size,
    'num_epochs' : args.num_epochs,
    'lr' : args.lr
    }
seed_everything(args.my_seed)

hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
hf_gpt2_tokenizer.pad_token = '[pad]'
GPT2_LM = TFGPT2LMHeadModel.from_pretrained('gpt2')
MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM_finetune' + '_epoch=' + str(kwargs['num_epochs']) + '_batch_size=' + str(kwargs['batch_size']) + '_lr=' + str(kwargs['lr'])
if os.path.exists(MODEL_SAVE_PATH):
    print(f"{MODEL_SAVE_PATH} -- Folder already exists \n")
else:
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    print(f"{MODEL_SAVE_PATH} -- Folder create complete \n")


# 학습 데이터 로드
train_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/input_sequence(train).npy')
eos_idx = indexing_eos_token(train_input_sequence)
train_input_sequence = train_input_sequence[np.where(eos_idx >= 4)[0], :]           # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

# 검증 데이터 로드
val_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/input_sequence(val).npy')
eos_idx = indexing_eos_token(val_input_sequence)
val_input_sequence = val_input_sequence[np.where(eos_idx >= 4)[0], :]               # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

# 학습 + 검증 데이터
full_input_sequence = np.concatenate([train_input_sequence, val_input_sequence], axis = 0)


# 허깅페이스 토큰에 맞춰 변환
with open('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer' + '/token_dict.pickle', 'rb') as f:
    token_dict = pickle.load(f)
special_token_list = ['[pad]', '[mask]']
add_token_list = special_token_list
token_dict = add_token_list_in_dict(add_token_list, token_dict)

# train_inputs_decoded = get_decoded_list(train_input_sequence, token_dict)               # (나의 토큰으로 토크나이징 된) train_input_sequence를 원형태로 디코딩
# val_inputs_decoded = get_decoded_list(val_input_sequence, token_dict)                   # (나의 토큰으로 토크나이징 된) val_input_sequence를 원형태로 디코딩
full_inputs_decoded = get_decoded_list(full_input_sequence, token_dict)

# 원형태로 디코딩 된 인풋 시퀀스를 허깅페이스 사전 기반 토크나이징을 통해 인코딩; padding = True를 통해 시퀀스 길이 맞춰주기
# train_inputs_txt = list(map(lambda x : " ".join(x), train_inputs_decoded))
# train_inputs_encoded = hf_gpt2_tokenizer(train_inputs_txt, return_tensors='tf', padding=True)
# val_inputs_txt = list(map(lambda x : " ".join(x), val_inputs_decoded))
# val_inputs_encoded = hf_gpt2_tokenizer(val_inputs_txt, return_tensors='tf', padding=True)
full_inputs_txt = list(map(lambda x : " ".join(x), full_inputs_decoded))
full_inputs_encoded = hf_gpt2_tokenizer(full_inputs_txt, return_tensors='tf', padding=True)


# 인풋 및 아웃풋 시퀀스로 나누기
# hf_train_inputs = train_inputs_encoded['input_ids'].numpy()
# train_inputs = hf_train_inputs[:, :hf_train_inputs.shape[1]-1]
# train_targets = hf_train_inputs[:, 1:hf_train_inputs.shape[1]]
# hf_val_inputs = val_inputs_encoded['input_ids'].numpy()
# val_inputs = hf_val_inputs[:, :hf_val_inputs.shape[1]-1]
# val_targets = hf_val_inputs[:, 1:hf_val_inputs.shape[1]]
hf_full_inputs = full_inputs_encoded['input_ids'].numpy()
full_inputs = hf_full_inputs[:, :hf_full_inputs.shape[1]-1]
full_targets = hf_full_inputs[:, 1:hf_full_inputs.shape[1]]


# Dataset 객체 자체는 cpu에 할당
with tf.device("/cpu:0"):
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)).shuffle(buffer_size = train_inputs.shape[0], reshuffle_each_iteration = False)
    # train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    # train_batchset = train_batchset.prefetch(1)

    # val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets)).shuffle(buffer_size = val_inputs.shape[0], reshuffle_each_iteration = False)
    # val_batchset = val_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    # val_batchset = val_batchset.prefetch(1)

    full_dataset = tf.data.Dataset.from_tensor_slices((full_inputs, full_targets)).shuffle(buffer_size = full_inputs.shape[0], reshuffle_each_iteration = False)
    full_batchset = full_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    full_batchset = full_batchset.prefetch(1)


optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])

'''
GPT2 파인튜닝 손실 함수
'''
# loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
def finetune_loss_function(real, pred):
    pad_token_id = hf_gpt2_tokenizer.pad_token_id

    # [PAD] 토큰들 (i.e., 값이 0인 token들)은 무시하는 mask 정의
    mask = tf.math.logical_not( tf.cast( tf.cast(tf.math.equal(real, pad_token_id), dtype = tf.int32), dtype = tf.bool ) )
    losses = sparse_categorical_cross_entropy(real, pred)                         # SparseCategoricalCrossentropy를 활용하여 loss함수 정의

    mask = tf.cast(mask, dtype = losses.dtype)
    losses *= mask

    sum_losses = tf.reduce_sum(losses, axis = 1)
    sum_mask = tf.reduce_sum(mask, axis = 1)

    return tf.reduce_mean(losses), tf.reduce_mean(sum_losses/sum_mask)

'''
GPT2 파인튜닝 정확도 함수
'''
def finetune_accuracy_function(real, pred):
    real = tf.cast(real, dtype = tf.int32)

    # 예측 토큰 반환
    max_pred = tf.argmax(pred, axis = -1)
    max_pred = tf.cast(tf.squeeze(max_pred), dtype = tf.int32)

    # 맞춘 토큰 행렬 (hit_matrix) 구축
    hit_index_mat = tf.cast(tf.where(real == max_pred), dtype = tf.int32)
    if len(hit_index_mat) == 0:
        num_hits = 0
    else:
        hit_matrix = tf.scatter_nd(hit_index_mat, tf.repeat(1, tf.shape(hit_index_mat)[0]), shape = tf.shape(real))
        num_hits = tf.reduce_sum(hit_matrix, axis = -1)            

    # padding 토큰 (token 0)에 대해서 masking된 행렬 구축
    pad_token_id = hf_gpt2_tokenizer.pad_token_id
    mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
    num_targets_without_padding = tf.reduce_sum(tf.cast(mask, dtype = tf.int32), axis = -1)

    # 각 sequence 별로 padding 제외 토큰들 중에서 맞춘 비율 계산
    acc = num_hits / num_targets_without_padding
    mean_acc = tf.reduce_mean(acc)

    return tf.cast(mean_acc, dtype = tf.float32)


'''
GPT2 파인튜닝 훈련함수
'''
@tf.function
def train_step(data, huggingface_model):

    inputs, targets = data

    with tf.GradientTape() as tape1:

        # 예측
        pred_outputs = huggingface_model(inputs, training = True)
        pred_logits = pred_outputs.logits

        # 손실 및 정확도 계산
        losses, _ = finetune_loss_function(targets, pred_logits)
        accuracies = finetune_accuracy_function(targets, pred_logits)

        # 최종 손실
        total_losses = losses

    # 최적화
    gradients = tape1.gradient(total_losses, huggingface_model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, huggingface_model.trainable_variables))

    return losses, accuracies

'''
GPT2 파인튜닝 검증함수
'''
@tf.function
def test_step(data, huggingface_model):

    inputs, targets = data

    # 예측
    pred_outputs = huggingface_model(inputs, training = False)
    pred_logits = pred_outputs.logits

    # 손실 및 정확도 계산
    losses, _ = finetune_loss_function(targets, pred_logits)
    accuracies = finetune_accuracy_function(targets, pred_logits)

    return losses, accuracies

# 훈련 메트릭
metrics_names = ['GPT2_loss', 'GPT2_acc']
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

    print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epochs']))
    # pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)
    pb_i = Progbar(len(full_batchset), stateful_metrics = metrics_names)

    # 훈련
    train_cumul_acc = val_cumul_acc = 0
    train_cumul_loss = val_cumul_loss = 0

    # 훈련 배치 루프
    # for idx, (train_inputs, train_targets) in enumerate(train_batchset):
    for idx, (train_inputs, train_targets) in enumerate(full_batchset):
        
        train_loss, train_acc = train_step((train_inputs, train_targets), GPT2_LM)

        # 배치별 정확도 및 손실 누계
        train_cumul_acc += train_acc.numpy()
        train_cumul_loss += train_loss.numpy()

        # 메트릭 값 업데이트
        metric_values = [('GPT2_loss', train_loss), ('GPT2_acc', train_acc)]
        pb_i.update(idx+1, values = metric_values)


    # 전체 평균 정확도 및 손실 (훈련셋)
    train_mean_acc = train_cumul_acc/(idx + 1)
    train_mean_loss = train_cumul_loss/(idx + 1)

    # 훈련 성능 출력
    train_acc_delta = train_mean_acc - train_pre_acc
    print('train_mean_loss : {}, train_mean_acc : {}, train_pre_acc : {}, train_acc_delta : {}'.format(train_mean_loss, train_mean_acc, train_pre_acc, train_acc_delta))
    train_pre_acc = train_mean_acc

    # # 검증 배치 루프
    # for idx, (val_inputs, val_targets) in enumerate(val_batchset):   
    #     val_loss, val_acc = test_step((val_inputs, val_targets), GPT2_LM)

    #     # 배치별 정확도 및 손실 누계
    #     val_cumul_acc += val_acc.numpy()
    #     val_cumul_loss += val_loss.numpy()

    # # 전체 평균 정확도 및 손실 (검증셋)
    # val_mean_loss = val_cumul_loss/(idx + 1)
    # val_mean_acc = val_cumul_acc/(idx + 1)

    # # 검증 성능 출력
    # val_acc_delta = val_mean_acc - val_pre_acc
    # print('val_mean_loss : {}, val_mean_acc : {}, val_pre_acc : {}, val_acc_delta : {}, max_val_pre_acc : {}'.format(val_mean_loss, val_mean_acc, val_pre_acc, val_acc_delta, max_val_pre_acc))
    # val_pre_acc = val_mean_acc


    # # 가중치 저장 조건
    # '''
    # validation_set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
    # '''
    # max_val_acc_delta = val_mean_acc - max_val_pre_acc
    # if max_val_acc_delta > 0.0:

    #     # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 이전 정확도 값 업데이트
    #     max_val_pre_acc = val_mean_acc

    #     # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 및 토크나이저 저장
    #     GPT2_LM.save_pretrained(MODEL_SAVE_PATH)
    #     # hf_gpt2_tokenizer.save_pretrained(MODEL_SAVE_PATH)    # 토크나이저는 저장을 안해야 pad_token_id가 50256으로 일치할 수도 ?


    # # 학습 중단 조건
    # '''
    # validation_set에 대해서 이전 k-epoch 동안 성능이 연속으로 저하되거나 훈련/검증 정확도 지표가 모두 0.999를 넘을 경우 경우 중단
    # '''
    # if len(patience_list) < k:
    #     patience_list += [val_acc_delta]
    # else:
    #     del patience_list[0]
    #     patience_list += [val_acc_delta]            
    # print('patience_list :', patience_list)
    # if len(np.where(np.array(patience_list) < 0)[0]) == k or (train_mean_acc + val_mean_acc) > (2 * 0.999):
    #     break;

    end_time = time.time()
    cur_sec = (end_time - start_time)%60
    cur_min = ((end_time - start_time)//60)%60
    cur_hr = ((end_time - start_time)//60)//60
    total_sec = (end_time - total_start_time)%60
    total_min = ((end_time - total_start_time)//60)%60
    total_hr = ((end_time - total_start_time)//60)//60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))


# 가중치 저장
GPT2_LM.save_pretrained(MODEL_SAVE_PATH)
