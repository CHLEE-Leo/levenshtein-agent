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
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

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
    'num_epochs' : args.num_epochs,
    'batch_size' : args.batch_size,
    'lr' : args.lr
    }
seed_everything(args.my_seed)
# kwargs = {
#     'num_epochs' : 10,
#     'batch_size' : 512,
#     'lr' : 5e-4
#     }
# seed_everything(920807)

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
checkpoint = "ydshieh/bert-base-uncased-yelp-polarity"

hf_bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
hf_bert_tokenizer.pad_token = '[pad]'
# distil_bert_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/Distil_BERT_lr=0'
distil_bert_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/yelp_Base_BERT_lr=0'

distil_BERT = TFAutoModelForSequenceClassification.from_pretrained(distil_bert_dir)

# MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/Distil_BERT' '_epoch=' + str(kwargs['num_epochs']) + '_batch_size=' + str(kwargs['batch_size']) + '_lr=' + str(kwargs['lr'])
MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/yelp_Base_BERT' '_epoch=' + str(kwargs['num_epochs']) + '_batch_size=' + str(kwargs['batch_size']) + '_lr=' + str(kwargs['lr'])

if os.path.exists(MODEL_SAVE_PATH):
    print(f"{MODEL_SAVE_PATH} -- Folder already exists \n")
else:
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    print(f"{MODEL_SAVE_PATH} -- Folder create complete \n")


# 학습 데이터 로드
train_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/input_sequence(train).npy')
eos_idx = indexing_eos_token(train_input_sequence)
train_input_sequence = train_input_sequence[np.where(eos_idx >= 4)[0], :]           # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
train_attribute = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/attribute(train).npy')
train_attribute = train_attribute[np.where(eos_idx >= 4)[0]]                          # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

# 검증 데이터 로드
val_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/input_sequence(val).npy')
eos_idx = indexing_eos_token(val_input_sequence)
val_input_sequence = val_input_sequence[np.where(eos_idx >= 4)[0], :]               # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
val_attribute = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/attribute(val).npy')
val_attribute = val_attribute[np.where(eos_idx >= 4)[0]]                              # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

# 허깅페이스 토큰에 맞춰 변환
with open('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer' + '/token_dict.pickle', 'rb') as f:
    token_dict = pickle.load(f)
special_token_list = ['[pad]', '[mask]']
add_token_list = special_token_list
token_dict = add_token_list_in_dict(add_token_list, token_dict)
train_inputs_decoded = get_decoded_list(train_input_sequence, token_dict)               # (나의 토큰으로 토크나이징 된) train_input_sequence를 원형태로 디코딩
val_inputs_decoded = get_decoded_list(val_input_sequence, token_dict)                   # (나의 토큰으로 토크나이징 된) val_input_sequence를 원형태로 디코딩

# 원형태로 디코딩 된 인풋 시퀀스를 허깅페이스 사전 기반 토크나이징을 통해 인코딩; padding = True를 통해 시퀀스 길이 맞춰주기
train_inputs_txt = list(map(lambda x : " ".join(x), train_inputs_decoded))
train_inputs_encoded = hf_bert_tokenizer(train_inputs_txt, return_tensors='tf', padding=True)
val_inputs_txt = list(map(lambda x : " ".join(x), val_inputs_decoded))
val_inputs_encoded = hf_bert_tokenizer(val_inputs_txt, return_tensors='tf', padding=True)

# 인풋 및 아웃풋 시퀀스로 나누기
hf_train_inputs = train_inputs_encoded['input_ids'].numpy()
train_inputs = copy.deepcopy(hf_train_inputs)

hf_val_inputs = val_inputs_encoded['input_ids'].numpy()
val_inputs = copy.deepcopy(hf_val_inputs)

# Dataset 객체 자체는 cpu에 할당
with tf.device("/cpu:0"):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_attribute)).shuffle(buffer_size = train_inputs.shape[0], reshuffle_each_iteration = False)
    train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    train_batchset = train_batchset.prefetch(1)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_attribute)).shuffle(buffer_size = val_inputs.shape[0], reshuffle_each_iteration = False)
    val_batchset = val_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    val_batchset = val_batchset.prefetch(1)

optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])

'''
distil_BERT 파인튜닝 손실 함수
'''
# loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
def finetune_loss_function(real, pred):

    # SparseCategoricalCrossentropy를 활용하여 loss함수 정의
    losses = sparse_categorical_cross_entropy(real, pred)

    return losses


'''
distil_BERT 파인튜닝 정확도 함수
'''
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_categorical_accuracy')
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_categorical_accuracy')
def finetune_accuracy_function(real, pred, mode):
    
    if mode == 'train':
        train_acc_metric.update_state(real, pred)
        mean_acc = train_acc_metric.result()
    elif mode == 'test':
        val_acc_metric.update_state(real, pred)
        mean_acc = val_acc_metric.result()

    return tf.cast(mean_acc, dtype = tf.float32)


'''
distil_BERT 파인튜닝 훈련함수
'''
@tf.function
def train_step(data, huggingface_model):

    inputs, labels = data

    with tf.GradientTape() as tape1:

        # 예측
        pred_outputs = huggingface_model(inputs, training = True)
        pred_logits = pred_outputs.logits

        # 손실 및 정확도 계산
        losses = finetune_loss_function(labels, pred_logits)
        accuracies = finetune_accuracy_function(labels, pred_logits, mode = 'train')

        # 최종 손실
        total_losses = losses

    # 최적화
    gradients = tape1.gradient(total_losses, huggingface_model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, huggingface_model.trainable_variables))

    return losses, accuracies

'''
distil_BERT 파인튜닝 검증함수
'''
@tf.function
def test_step(data, huggingface_model):

    inputs, labels = data

    # 예측
    pred_outputs = huggingface_model(inputs, training = False)
    pred_logits = pred_outputs.logits

    # 손실 및 정확도 계산
    losses = finetune_loss_function(labels, pred_logits)
    accuracies = finetune_accuracy_function(labels, pred_logits, mode = 'test')

    return losses, accuracies

# 훈련 메트릭
metrics_names = ['distilBERT_loss', 'distilBERT_acc']
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
    pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)

    # 훈련
    train_cumul_acc = val_cumul_acc = 0
    train_cumul_loss = val_cumul_loss = 0

    # 훈련 배치 루프
    for idx, (train_inputs, train_labels) in enumerate(train_batchset):        
        
        train_loss, train_acc = train_step((train_inputs, train_labels), distil_BERT)

        # 배치별 정확도 및 손실 누계
        train_cumul_acc += train_acc.numpy()
        train_cumul_loss += train_loss.numpy()

        # 메트릭 값 업데이트
        metric_values = [('distilBERT_loss', train_loss), ('distilBERT_acc', train_acc)]
        pb_i.update(idx+1, values = metric_values)


    # 전체 평균 정확도 및 손실 (훈련셋)
    train_mean_acc = train_cumul_acc/(idx + 1)
    train_mean_loss = train_cumul_loss/(idx + 1)

    # 훈련 성능 출력
    train_acc_delta = train_mean_acc - train_pre_acc
    print('train_mean_loss : {}, train_mean_acc : {}, train_pre_acc : {}, train_acc_delta : {}'.format(train_mean_loss, train_mean_acc, train_pre_acc, train_acc_delta))
    train_pre_acc = train_mean_acc


    # 검증 배치 루프
    for idx, (val_inputs, val_labels) in enumerate(val_batchset):   
        val_loss, val_acc = test_step((val_inputs, val_labels), distil_BERT)

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

        # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 및 토크나이저 저장
        distil_BERT.save_pretrained(MODEL_SAVE_PATH)
        # hf_gpt2_tokenizer.save_pretrained(MODEL_SAVE_PATH)    # 토크나이저는 저장을 안해야 pad_token_id가 50256으로 일치할 수도 ?


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
