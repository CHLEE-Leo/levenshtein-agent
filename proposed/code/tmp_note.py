# %%
import pickle
import tensorflow as tf
import numpy as np
from proposed.autoencoder.utils import fill_pad_after_eos, get_token

# 단어 사진 가져오기
with open('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer' + '/token_dict.pickle', 'rb') as f:
    token_dict = pickle.load(f)

# 모델 파라미터 지정
model_name = 'aetransformer'
kwargs = {
    'model_name' : model_name,
    'batch_size' : 512,
    'lr' : 1e-5,
    'num_layers_enc' : 2,
    'num_layers_dec' : 2,
    'num_layers_rnn' : 1,
    'num_layers_cla' : 1,
    'd_model' : 0,
    'd_model_enc' : 256,
    'd_model_dec' : 256,
    'd_model_rnn' : 256,
    'd_model_cla' : 256,
    'num_heads' : 0,
    'num_heads_enc' : 8,
    'num_heads_dec' : 8,
    'num_heads_rnn' : 4,
    'd_ff' : 1024,
    'rnn_module' : 'gru',
    'head_type' : 'single',
    'cla_dropout' : 0.1,
    'cla_regularization' : 'orthonormal',
    'mode' : 'inference',
    'dropout_rate' : 0.1,
    'vocab_size' : len(token_dict),
    'num_epochs' : 100,
    'num_attribute' : 2,
    'epsilon' : 1e-2
}

# (1) 데이터 로드
gen_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer/input_sequence(gen_train).npy')
input_sequences = copy.deepcopy(gen_input_sequence[:, :gen_input_sequence.shape[1]-1])
output_sequences = copy.deepcopy(gen_input_sequence[:, :gen_input_sequence.shape[1]-1])
target_sequences = gen_input_sequence[:, 1:gen_input_sequence.shape[1]]
polarity = np.load('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer/polarity(gen_train).npy')

# Dataset 객체 자체는 cpu에 할당
with tf.device("/cpu:0"):
    # dataset = tf.data.Dataset.from_tensor_slices((input_sequences, output_sequences, target_sequences, polarity)).shuffle(buffer_size = input_sequences.shape[0], reshuffle_each_iteration = True)
    dataset = tf.data.Dataset.from_tensor_slices((input_sequences, output_sequences, target_sequences, polarity)).shuffle(buffer_size = input_sequences.shape[0], reshuffle_each_iteration = False)
    batchset = dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    batchset = batchset.prefetch(1)

model = AETransformer(**kwargs)
inputs, outputs, targets, attributes = list(batchset)[0]
model((inputs, outputs))
# %%
# --------------------------------------------------------------------------------------------------------------------------------------
for i in range(len(model.decoder.stacked_dec_layers[0].weights)):
    weight_name = model.decoder.stacked_dec_layers[0].weights[i].name.split('/')[-1]
    if 'kernel' in weight_name:
        print('{}-th layers is kernel'.format(i))
# %%
# --------------------------------------------------------------------------------------------------------------------------------------
kernels = tf.zeros(shape = (256, 0))
for i in range(len(model.decoder.stacked_dec_layers[0].weights)):
    weight_name = model.decoder.stacked_dec_layers[0].weights[i].name.split('/')[-1]
    if 'kernel' in weight_name:
        a_kernel = model.decoder.stacked_dec_layers[0].weights[i]
        if a_kernel.shape == (256, 256):
            kernels = tf.concat([kernels, a_kernel], axis = -1)
# %%
# --------------------------------------------------------------------------------------------------------------------------------------
# Reward Quantiles 구하는 코드
a = [0.5, 0.1, 0.7, 0.3, 0.9]
print(a)
sorted_idx = tf.argsort(a)
sorted_a = tf.gather(a, sorted_idx)
softmax_sorted_a = tf.math.exp(sorted_a)/tf.reduce_sum(tf.math.exp(sorted_a))
a_cdf = tf.math.cumsum(softmax_sorted_a).numpy()
target_tau = 0.70
target_val = sorted_a[a_cdf >= target_tau]
target_val_idx = tf.where(tf.math.equal(a, tf.reshape(target_val, shape = (-1, 1))) == True)[:, -1]
tf.gather(a, target_val_idx)

# %%
# --------------------------------------------------------------------------------------------------------------------------------------
# 보상 추정기 초기화
reward_function = Reward_Estimator(**kwargs)

# --> 사전학습된 보상함수 로드
load_dir = '/home/messy92/Leo/NAS_folder/ICLR22/weights/text-style-transfer/reward_estimator_reward_function_500'
print(load_dir)
reward_function.load_weights(tf.train.latest_checkpoint(load_dir))

for idx, (inputs, attributes) in enumerate(list(batchset)[:2]):        
    enc_pad_mask, _, _ = reward_function.mask_generator(inputs, inputs)

    # 예측
    pred_reward = reward_function(inputs, enc_pad_mask)


num_quantiles = 5
per_quantile_size = int( np.ceil( len(sorted_reward_vec)/num_quantiles ) )
per_quantile_range_idx = [per_quantile_size * i for i in range(num_quantiles + 1)]

pre_quantile_idx = 0
for i in range(1, num_quantiles + 1):
    quantile_idx = per_quantile_range_idx[i]
    per_quantile_reward_vec = sorted_reward_vec.numpy()[pre_quantile_idx:quantile_idx]
    print(per_quantile_reward_vec)
    pre_quantile_idx = quantile_idx + 1

    # per_quantile_reward_vec의 값들의 reward_vec 내에서의 위치값 반환
    target_idx = np.where(np.in1d(reward_vec, per_quantile_reward_vec) == True)[0]

    print(tmp)


# %%
# --------------------------------------------------------------------------------------------------------------------------------------
# 생성과정 실험 코드

# 학습 데이터 로드
test_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer/input_sequence(test).npy')
test_input_sequences = copy.deepcopy(test_input_sequence[:, :test_input_sequence.shape[1]-1])
test_output_sequences = copy.deepcopy(test_input_sequence[:, :test_input_sequence.shape[1]-1])
test_target_sequences = test_input_sequence[:, 1:test_input_sequence.shape[1]]
test_polarity = np.load('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer/polarity(test).npy')

# 토큰 사전 가져오기
with open('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer' + '/token_dict.pickle', 'rb') as f:
    token_dict = pickle.load(f)
special_token_list = ['[pad]', '[mask]']
edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
add_token_list = special_token_list + edit_token_list
token_dict = add_token_list_in_dict(add_token_list, token_dict)
action_set = list(token_dict.values())[-len(edit_token_list):]

model_name = 'Mask_LM'
kwargs = {
    'model_name' : model_name,
    'batch_size' : 1024,
    'lr' : 5 * 1e-5,
    'num_layers_enc' : 6,
    'num_layers_dec' : 6,
    'd_model' : 0,
    'd_model_enc' : 512,
    'd_model_dec' : 512,
    'dec_masking_window_len' : None,
    'num_heads' : 0,
    'num_heads_enc' : 4,
    'num_heads_dec' : 4,
    'd_ff' : 1024,
    'stack': 'rnn',
    'dropout_rate' : 0.1,
    'vocab_size' : len(token_dict),
    'num_epochs' : 500,
    'action_space' : action_set
}

target_agent_both = AETransformer(**kwargs)
load_dir2 = '/home/messy92/Leo/NAS_folder/ICLR22/weights/text-style-transfer/BART_ST_both_rnn_500_w=None'
target_agent_both.load_weights(tf.train.latest_checkpoint(load_dir2))

target_agent_pos = AETransformer(**kwargs)
load_dir2 = '/home/messy92/Leo/NAS_folder/ICLR22/weights/text-style-transfer/BART_ST_pos_rnn_500'
target_agent_pos.load_weights(tf.train.latest_checkpoint(load_dir2))

target_agent_neg = AETransformer(**kwargs)
load_dir2 = '/home/messy92/Leo/NAS_folder/ICLR22/weights/text-style-transfer/BART_ST_neg_rnn_500'
target_agent_neg.load_weights(tf.train.latest_checkpoint(load_dir2))

pos_test_input_sequence = copy.deepcopy(test_input_sequences[-3, :][np.newaxis, :])
pos_test_output_sequence = copy.deepcopy(test_output_sequences[-3, :][np.newaxis, :])
neg_test_input_sequence = copy.deepcopy(test_input_sequences[7, :][np.newaxis, :])
neg_test_output_sequence = copy.deepcopy(test_output_sequences[7, :][np.newaxis, :])
print('gold_pos_seqs :', [ token_dict[t] for t in pos_test_input_sequence[0, :] ])
print('gold_neg_seqs :', [ token_dict[t] for t in neg_test_input_sequence[0, :] ])

# reward_function에 스타일 전환된 시퀀스를 입력
enc_pad_mask1, _, _ = reward_function.mask_generator(pos_test_input_sequence, pos_test_input_sequence)
enc_pad_mask2, _, _ = reward_function.mask_generator(neg_test_input_sequence, neg_test_input_sequence)
reward1 = tf.nn.softmax(reward_function(pos_test_input_sequence, enc_pad_mask1, training = False), axis = -1)
reward2 = tf.nn.softmax(reward_function(neg_test_input_sequence, enc_pad_mask2, training = False), axis = -1)
print('reward 1 : {}, reward 2 : {}'.format(reward1, reward2))

# reward_function에 스타일 전환된 시퀀스를 입력 (토탈)
enc_pad_mask, _, _ = reward_function.mask_generator(test_input_sequences, test_input_sequences)
reward0 = tf.nn.softmax(reward_function(test_input_sequences, enc_pad_mask, training = False), axis = -1)
print('mean reward 0 : {}'.format(tf.reduce_mean(reward0[:, 1])))


# # [pad] 갯수만큼 앞에 mask_span 추가해보기
# num_zeros = tf.where(pos_test_input_sequence == 0).shape[0]
# pos_test_input_sequence1 = tf.gather_nd(pos_test_input_sequence, tf.where( tf.math.logical_and(pos_test_input_sequence != 0, pos_test_input_sequence != 2) ))
# pos_test_input_sequence1 = tf.concat([ tf.cast(tf.repeat( get_token(token_dict, '[mask]') , num_zeros-1 ), dtype = tf.int32), pos_test_input_sequence1], axis = -1)[tf.newaxis, :]
# pos_test_input_sequence1 = pos_test_input_sequence1.numpy()

# num_zeros = tf.where(neg_test_input_sequence == 0).shape[0]
# neg_test_input_sequence1 = tf.gather_nd(neg_test_input_sequence, tf.where( tf.math.logical_and(neg_test_input_sequence != 0, neg_test_input_sequence != 2) ))
# neg_test_input_sequence1 = tf.concat([ tf.cast(tf.repeat( get_token(token_dict, '[mask]') , num_zeros-1 ), dtype = tf.int32), neg_test_input_sequence1], axis = -1)[tf.newaxis, :]
# neg_test_input_sequence1 = neg_test_input_sequence1.numpy()

# pos_test_input_sequence1 = tf.concat([ tf.constant([2], dtype = tf.int32)[:, tf.newaxis], pos_test_input_sequence1 ], axis= -1 ).numpy()
# neg_test_input_sequence1 = tf.concat([ tf.constant([2], dtype = tf.int32)[:, tf.newaxis], neg_test_input_sequence1 ], axis = -1).numpy()

# print('gold_pos_seqs :', [ token_dict[t] for t in pos_test_input_sequence1[0, :] ])
# print('gold_neg_seqs :', [ token_dict[t] for t in neg_test_input_sequence1[0, :] ])


# 직접 특정 포지션에 mask_span 삽입하기
print('\n')
pos_test_input_sequence[0, np.array([12, 14])] = get_token(token_dict, '[mask]')
pos_test_output_sequence[0, np.array([12, 14])] = get_token(token_dict, '[mask]')
neg_test_input_sequence[0, np.array([1, 2, 5, 6])] = get_token(token_dict, '[mask]')
neg_test_output_sequence[0, np.array([1, 2, 5, 6])] = get_token(token_dict, '[mask]')
# pos_test_input_sequence[0, np.arange(4, pos_test_input_sequence.shape[1])] = get_token(token_dict, '[mask]')
# pos_test_output_sequence[0, np.arange(4, pos_test_output_sequence.shape[1])] = get_token(token_dict, '[mask]')
# neg_test_input_sequence[0, np.arange(2, neg_test_input_sequence.shape[1])] = get_token(token_dict, '[mask]')
# neg_test_output_sequence[0, np.arange(2, neg_test_output_sequence.shape[1])] = get_token(token_dict, '[mask]')
print('gold_pad_seqs :', [ token_dict[t] for t in pos_test_input_sequence[0, :] ])
print('gold_pad_seqs :', [ token_dict[t] for t in neg_test_input_sequence[0, :] ])

# gen_seqs_pos_to_neg = bart_neg.inference((pos_test_input_sequence, pos_test_output_sequence), decoding='top_k', top_k = 5)
# gen_seqs_neg_to_pos = bart_pos.inference((neg_test_input_sequence, neg_test_output_sequence), decoding='top_k', top_k = 5)
gen_seqs_both_pos = target_agent_both.inference((pos_test_input_sequence, pos_test_output_sequence), token_dict, decoding='top_k', top_k = 5)
gen_seqs_both_neg = target_agent_both.inference((neg_test_input_sequence, neg_test_input_sequence), token_dict, decoding='top_k', top_k = 5)
print('pos :', [ token_dict[t] for t in gen_seqs_both_pos.numpy()[0, :] ])
print('neg :', [ token_dict[t] for t in gen_seqs_both_neg.numpy()[0, :] ])
# gen_seqs_pos_to_neg = target_agent_neg.inference((pos_test_input_sequence1, pos_test_input_sequence1), decoding='greedy', top_k = 5)    # mask_span이 추가된 경우
# gen_seqs_neg_to_pos = target_agent_pos.inference((neg_test_input_sequence1, neg_test_input_sequence1), decoding='greedy', top_k = 5)    # mask_span이 추가된 경우
# print('pos_to_neg :', [ token_dict[t] for t in gen_seqs_pos_to_neg.numpy()[0, :] ])
# print('neg_to_pos :', [ token_dict[t] for t in gen_seqs_neg_to_pos.numpy()[0, :] ])

# reward_function에 스타일 전환된 시퀀스를 입력
enc_pad_mask1, _, _ = reward_function.mask_generator(gen_seqs_pos_to_neg, gen_seqs_pos_to_neg)
enc_pad_mask2, _, _ = reward_function.mask_generator(gen_seqs_neg_to_pos, gen_seqs_neg_to_pos)
reward1 = tf.nn.softmax(reward_function(gen_seqs_pos_to_neg, enc_pad_mask1, training = False), axis = -1)
reward2 = tf.nn.softmax(reward_function(gen_seqs_neg_to_pos, enc_pad_mask2, training = False), axis = -1)
print('reward 1 : {}, reward 2 : {}'.format(reward1, reward2))


for k in range(10):
    print('\n')
    print('gen_seqs_pos :', [ token_dict[t] for t in gen_seqs_pos_to_neg.numpy() ])
    print('gen_seqs_neg :', [ token_dict[t] for t in gen_seqs_neg_to_pos.numpy() ])

# i = 0
# j = 10
# gen_seqs = target_agent.inference((test_input_sequences[i:j, :], test_output_sequences[i:j, :]), decoding='top_k', top_k = 50)
# for k in range(10):
#     print('\n')
#     print('gold_seqs :', [ token_dict[t] for t in test_target_sequences[(i+k), :] ])
#     print('gen_seqs :', [ token_dict[t] for t in gen_seqs.numpy()[k, :] ])

import numpy as np
a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
a2 = np.array([[4,5,6],[7,8,9],[1,1,1]])
a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

# %%
## CANVAS 코드
# action 셋 
action_set = dict({
                    0 : '[mask]',
                    1 : '[bos]', 2 : '[eos]',
                    3 : '[INS_F]', 4 : '[INS_B]', 5 : '[INS_A]',
                    6 : '[DEL]',
                    7 : '[REP]',
                    8 : '[KEP]'
                    })

tmp_batch = np.array([
                        [2, 9577, 9582, 9582, 9577, 9581, 3, 0, 0, 0, 0], 
                        [2, 9581, 9582, 9579, 9578, 9580, 3, 0, 0, 0, 0], 
                        [2, 9581, 9582, 9577, 9580, 9578, 3, 0, 0, 0, 0]                        
                    ])
# print([token_dict[t] for t in tmp_batch[0, :]])
# print([token_dict[t] for t in tmp_batch[1, :]])
# print([token_dict[t] for t in tmp_batch[2, :]])

# 캔버스 만들기
tmp_batch = np.array([[2, 10, 20, 30, 100, 200, 3, 0, 0, 0, 0], [2, 40, 50, 60, 100, 3, 0, 0, 0, 0, 0]])
print([token_dict[t] for t in tmp_batch[0, :]])
print([token_dict[t] for t in tmp_batch[1, :]])

tmp_batch_prime = tf.ones(shape = (tmp_batch.shape[0], 2 * tmp_batch.shape[1] - 1))

# pad_idx = tf.cast(tf.where(tmp_batch == 0), dtype = tf.int32)

value_fix_idx = tf.cast(tf.where(1 - tf.range(tf.shape(tmp_batch_prime)[1]) % 2), dtype = tf.int32)
col_idx = tf.tile(value_fix_idx, multiples = (tf.shape(tmp_batch)[0], 1))
row_idx = tf.cast(tf.repeat(tf.range(tf.shape(tmp_batch)[0]), tf.shape(tmp_batch)[1])[:, tf.newaxis], dtype = tf.int32)
batch_idx = tf.concat([row_idx, col_idx], axis = -1)

# 제로 버퍼 캔버스
col_shape = tf.constant(2 * tf.shape(tmp_batch)[1] - 1).numpy()
row_shape = tf.shape(tmp_batch)[0].numpy()
target_shape = (row_shape, col_shape)
aug_batch = tf.cast(tf.scatter_nd(indices = batch_idx, updates = tf.squeeze(tf.reshape(tmp_batch, shape = (-1, 1))), shape = target_shape), dtype = tf.int32)

# 1) INS 오퍼레이션 시도 처리
canvas_with_zero = aug_batch
ins_f_token = get_token(token_dict, '[INS_B]')          # [INS_F]에 해당하는 토큰
ins_f_idx = tf.where(canvas_with_zero == ins_f_token)   # [INS_F]에 해당하는 토큰의 인덱스

add_rep_idx = ins_f_idx.numpy()                         # [INS_F]의 한 시점 앞에 추가로 [REP]가 삽입될 위치
add_rep_idx[:, 1] = add_rep_idx[:, 1] + 1                       

target_val = tf.gather_nd(canvas_with_zero, indices = add_rep_idx)
rep_token = tf.repeat([get_token(token_dict, '[REP]')], repeats = tf.shape(target_val))             # [REP]에 해당하는 토큰

canvas_with_ins_f = tf.tensor_scatter_nd_update(tensor = canvas_with_zero, indices = add_rep_idx, updates = rep_token)
print([token_dict[t] for t in canvas_with_ins_f.numpy()[0, :]])
print([token_dict[t] for t in canvas_with_ins_f.numpy()[1, :]])
print([token_dict[t] for t in canvas_with_ins_f.numpy()[2, :]])

# eos 토큰 이후의 토큰들 (= [pad] 토큰들의 row/col 인덱싱 해주기)
canvas_len = 2 * tf.shape(tmp_batch)[1] - 1
canvas_len_vec = tf.repeat(canvas_len, tf.shape(tmp_batch)[0])
canvas_eos_idx = tf.argmax(tf.math.cumsum(aug_batch, axis = -1), axis = -1)

first_pad_idx = canvas_eos_idx + 1
pad_idx_len = canvas_len - tf.cast(first_pad_idx, dtype = tf.int32)

pad_col_idx_tile = np.concatenate(tf.ragged.range(first_pad_idx, canvas_len_vec).to_list())[:, tf.newaxis]
pad_row_idx_tile = tf.repeat(tf.range(tf.shape(tmp_batch)[0]), pad_idx_len)[:, tf.newaxis]
pad_batch_idx = tf.concat([pad_row_idx_tile, pad_col_idx_tile], axis = -1)

# [mask] 넣어주기
target_idx = tf.cast(tf.where(aug_batch == 0), dtype = tf.int32)
flatten_mask_vector = tf.ones(target_idx.shape[0], dtype = tf.int32) * get_token(token_dict, '[mask]')
aug_batch = tf.tensor_scatter_nd_update(aug_batch, indices = target_idx, updates = flatten_mask_vector)

# [pad] 넣어주기
flatten_pad_vector = tf.zeros(shape = tf.shape(pad_batch_idx)[0], dtype = tf.int32)
tf.tensor_scatter_nd_update(tensor = aug_batch, indices = pad_batch_idx, updates = flatten_pad_vector)


print('gold_pos_seqs :', [ token_dict[t] for t in test_input_sequence[0, :] ])
ddd = create_canvas(test_input_sequence, token_dict)
print('gold_pos_seqs :', [ token_dict[t] for t in ddd.numpy()[0, :] ])

# %%
'''
일단, 매 time-step마다 [INS_f], [INS_b], [INS_a], [DEL], [REP], [KEP] 중 하나를 선택하는 레번슈타인 오버레이션 신경망 짜주기.
# [INS_f] = Insert front
# [INS_b] = Insert behind
# [INS_a] = Insert all
이 신경망의 인풋은 토큰 시퀀스임.
이 신경망의 아웃풋은 액션 시퀀스임.

1)
액션 시퀀스를 위의 코드로 사이사이에 [unk] 토큰이 들어간 캔버스 형태로 확장해주기.
토큰 시퀀스를 위의 코드로 사이사이에 [unk] 토큰이 들어간 캔버스 형태로 확장해주기.

2)
캔버스 액션 시퀀스에서 [INS_F]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 t-1에 해당하는 곳에 [REP] 넣어주기
캔버스 액션 시퀀스에서 [INS_B]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 t+1에 해당하는 곳에 [REP] 넣어주기
캔버스 액션 시퀀스에서 [INS_A]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 t-1, t+1에 해당하는 곳에 [REP] 넣어주기

3)
캔버스 액션 시퀀스에서 [INS]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 [KEP]으로 변경

4)
기존의 캔버스와 동일한 shape를 가지되, [DEL]에 의한 토큰 삭제가 반영된 캔버스 생성하기

--> 캔버스 액션 시퀀스에서 [DEL]과 [PAD]를 제외한 모든 다른 액션 (= [REP]과 [KEP]만 남음)들의 batch_idx를 뽑기
--> batch_idx의 col_idx를 각 row_idx별로 0부터 오름차순으로 재할당하기
--> 아래의 코드 수행하기
taget_val = tf.gather_nd(tmp_batch, indices = batch_idx)
new_tmp_batch = tf.scatter_nd(indices = batch_idx, updates = target_val, shape = tf.shape(tmp_batch))


각 액션 시퀀스 별로 [DEL]의 갯수를 뽑기
뽑힌 [DEL]의 갯수만큼 각 액션 시퀀스에 [PAD] 추가해주기

'''
# %%
'''
LevAgent (LEVA) 코드 실험
'''
# target_idx = np.random.choice(a = np.arange(test_input_sequences.shape[0]), size = 32, replace=False)
# inputs = test_input_sequences[target_idx, :]
# polars = np.load('/home/messy92/Leo/NAS_folder/ICLR22/prep_data/text-style-transfer/polarity(gen_test).npy')
# polars = polars[target_idx]

target_idx = np.random.choice(a = np.arange(gen_input_sequence.shape[0]), size = 32, replace=False)
inputs = gen_input_sequence[target_idx, :]
polars = polarity[target_idx]
print('\n')
print([token_dict[t] for t in inputs[0, :]])

enc_pad_mask, _, _ = bart_pos.mask_generator(inputs, inputs)

# 예측
agent_outputs, action_att_weights = lev_agent(inputs, enc_pad_mask)
agent_actions, action_tokens = return_action_token(agent_outputs, token_dict)   # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)

# 레반슈타인 연산자 적용
masked_inputs = apply_lev_operation(inputs, action_tokens, token_dict)

# masked_inputs을 pos_masked_inputs & neg_masked_inputs으로 분리
pos_masked_inputs, pos_idx, neg_masked_inputs, neg_idx = split_by_polarity(masked_inputs[:, :masked_inputs.shape[1]-1], polars)
print('\n')
my_idx = 7
print([token_dict[t] for t in inputs[pos_idx[my_idx], :]])
print([token_dict[t] for t in pos_masked_inputs[my_idx, :]])
print('\n')
print([token_dict[t] for t in inputs[neg_idx[my_idx], :]])
print([token_dict[t] for t in neg_masked_inputs[my_idx, :]])

# masked_inputs_pos은 bart_neg 모델에, masked_inputs_neg은 bart_pos에 입력
pos_to_neg_outputs = bart_neg.inference((pos_masked_inputs, pos_masked_inputs))
neg_to_pos_outputs = bart_pos.inference((neg_masked_inputs, neg_masked_inputs))
print('\n')
print([token_dict[t] for t in pos_to_neg_outputs[my_idx, :].numpy()])
print([token_dict[t] for t in neg_to_pos_outputs[my_idx, :].numpy()])

pos_to_neg_outputs = fill_pad_after_eos(pos_to_neg_outputs, token_dict)
neg_to_pos_outputs = fill_pad_after_eos(neg_to_pos_outputs, token_dict)
print('\n')
print([token_dict[t] for t in pos_to_neg_outputs[my_idx, :].numpy()])
print([token_dict[t] for t in neg_to_pos_outputs[my_idx, :].numpy()])

# reward_function에 스타일 전환된 시퀀스를 입력
enc_pad_mask1, _, _ = reward_function.mask_generator(pos_to_neg_outputs, pos_to_neg_outputs)
enc_pad_mask2, _, _ = reward_function.mask_generator(neg_to_pos_outputs, neg_to_pos_outputs)
reward1 = tf.nn.softmax(reward_function(pos_to_neg_outputs, enc_pad_mask1, training = False), axis = -1)
reward2 = tf.nn.softmax(reward_function(neg_to_pos_outputs, enc_pad_mask2, training = False), axis = -1)
print('reward 1 : {}, reward 2 : {}'.format(reward1[my_idx], reward2[my_idx]))

# reward_function에 기존 시퀀스를 입력
enc_pad_mask1, _, _ = reward_function.mask_generator(inputs[pos_idx, :], inputs[pos_idx, :])
enc_pad_mask2, _, _ = reward_function.mask_generator(inputs[neg_idx, :], inputs[neg_idx, :])
reward1 = tf.nn.softmax(reward_function(inputs[pos_idx, :], enc_pad_mask1, training = False), axis = -1)
reward2 = tf.nn.softmax(reward_function(inputs[neg_idx, :], enc_pad_mask2, training = False), axis = -1)
print('reward 1 : {}, reward 2 : {}'.format(reward1[my_idx], reward2[my_idx]))

enc_pad_mask1, _, _ = reward_function.mask_generator(gen_input_sequence[:32, :], gen_input_sequence[:32, :])
reward1 = tf.nn.softmax(reward_function(gen_input_sequence[:32, :], enc_pad_mask1, training = False), axis = -1)
print('reward 1 : {}'.format(reward1[my_idx]))

# %%
enc_pad_mask1, _, _ = reward_function.mask_generator(inputs[:1, :], inputs[:1, :])
reward1 = tf.nn.softmax(reward_function(inputs[:1, :], enc_pad_mask1, training = False), axis = -1)
print('reward 1 : {}'.format(reward1))


# %%
pos_test_input_sequence = copy.deepcopy(inputs[3, :][np.newaxis, :])
neg_test_input_sequence = copy.deepcopy(inputs[2, :][np.newaxis, :])
print('gold_pos_seqs :', [ token_dict[t] for t in pos_test_input_sequence[0, :] ])
print('gold_neg_seqs :', [ token_dict[t] for t in neg_test_input_sequence[0, :] ])

# 직접 특정 포지션에 mask_span 삽입하기
print('\n')
pos_test_input_sequence[0, np.array([3])] = get_token(token_dict, '[mask]')
neg_test_input_sequence[0, np.array([1, 2, 5, 6, 7])] = get_token(token_dict, '[mask]')
# pos_test_input_sequence[0, np.arange(4, pos_test_input_sequence.shape[1])] = get_token(token_dict, '[mask]')
# pos_test_output_sequence[0, np.arange(4, pos_test_output_sequence.shape[1])] = get_token(token_dict, '[mask]')
# neg_test_input_sequence[0, np.arange(2, neg_test_input_sequence.shape[1])] = get_token(token_dict, '[mask]')
# neg_test_output_sequence[0, np.arange(2, neg_test_output_sequence.shape[1])] = get_token(token_dict, '[mask]')
print('gold_pad_seqs :', [ token_dict[t] for t in pos_test_input_sequence[0, :] ])
print('gold_pad_seqs :', [ token_dict[t] for t in neg_test_input_sequence[0, :] ])


gen_seqs_pos_to_neg = target_agent_neg.inference((pos_test_input_sequence, pos_test_output_sequence), decoding='top_k', top_k = 5)
gen_seqs_neg_to_pos = target_agent_pos.inference((neg_test_input_sequence, neg_test_output_sequence), decoding='top_k', top_k = 5)
# gen_seqs_pos_to_neg = target_agent_neg.inference((pos_test_input_sequence1, pos_test_input_sequence1), decoding='greedy', top_k = 5)    # mask_span이 추가된 경우
# gen_seqs_neg_to_pos = target_agent_pos.inference((neg_test_input_sequence1, neg_test_input_sequence1), decoding='greedy', top_k = 5)    # mask_span이 추가된 경우
print('pos_to_neg :', [ token_dict[t] for t in gen_seqs_pos_to_neg.numpy()[0, :] ])
print('neg_to_pos :', [ token_dict[t] for t in gen_seqs_neg_to_pos.numpy()[0, :] ])

# %%
'''
MSE 손실함수 실험
'''
reals = np.array([-1., 1., 1., -1.])
preds = np.array([-1., -1., -1., -1.])
# print(tf.norm(reals - preds))
mse_loss = tf.keras.losses.MeanSquaredError(name = 'mean_squared_error')
print(mse_loss(reals, preds))
reals = np.array([0, 1, 1, 0])
binary_accuracy = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold = 0.0)
binary_accuracy(tf.expand_dims(reals, axis = 1), tf.expand_dims(preds, axis = 1))





#%%
kwargs = {'batch_size' : 32,
            'kernel_size_list' : [3, 4, 5],
            'stride_size' : 3,
            'd_model' : 256,
            'num_filters' : 3,
            'max_seq_len' : 18
           }
tmp_mf_CNN = Multi_KernelSize_CNN(**kwargs)

tmp_embeds = tf.random.normal(shape = (32, 18, 256, 1))
aaa = tmp_mf_CNN(tmp_embeds)
bbb(tf.transpose(tf.squeeze(aaa), perm = (0, 2, 1)))



# %%
# LEVA 코드 중간에 실시간으로 생성되는 문장들 보는 코드

# 실험!!!!!!!!!!!!!!!!
rand_idx = tf.random.uniform(shape = (32, ), minval = 0, maxval = train_input_sequence.shape[0], dtype = tf.int32).numpy()
train_input_sequence = train_input_sequence[rand_idx, :]
train_polarity = train_polarity[rand_idx]


my_idx = 0
# print('polars : {} and my_polar : {}'.format(polars, polars[my_idx]))
# print('pos_idx : {} and neg_idx : {}'.format(np.where(polars == 1)[0], np.where(polars == 0)[0]))
my_pos_idx = np.where(polars == 1)[0][my_idx]
my_neg_idx = np.where(polars == 0)[0][my_idx]
print('\n')
print([token_dict[t] for t in inputs[my_pos_idx, :].numpy()])
print([token_dict[t] for t in pos_to_neg_outputs[my_idx, :].numpy()])
print('\n')
print([token_dict[t] for t in inputs[my_neg_idx, :].numpy()])
print([token_dict[t] for t in neg_to_pos_outputs[my_idx, :].numpy()])


# %%
from official import nlp
from official.nlp.modeling.ops import sampling_module
from official.nlp.modeling.ops import beam_search

params = {}
params['num_heads'] = 2
params['num_layers'] = 2
params['batch_size'] = 2
params['n_dims'] = 256
params['max_decode_length'] = 4

cache = {
    'layer_%d' % layer: {
        'k': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], int(params['n_dims']/params['num_heads'])], dtype=tf.float32),
        'v': tf.zeros([params['batch_size'], params['max_decode_length'], params['num_heads'], int(params['n_dims']/params['num_heads'])], dtype=tf.float32)
        } for layer in range(params['num_layers'])
    }
print("cache key shape for layer 1 :", cache['layer_1']['k'].shape)

probabilities = tf.constant([[[0.3, 0.4, 0.3], [0.3, 0.3, 0.4],
                              [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                            [[0.2, 0.5, 0.3], [0.2, 0.7, 0.1],
                              [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]])

def length_norm(length, dtype):
    """Return length normalization factor."""
    return tf.pow(((5. + tf.cast(length, dtype)) / 6.), 0.0)

'''
logits을 반환하는 함수 --> model_fn(i) 함수 안에 Transformer 디코더 모델을 써야할 듯?
-- i : time_step 인덱스
'''
def model_fn(i):
    return probabilities[:, i, :]

'''
symbols (아마 시퀀스?)을 logits으로 반환하는 함수
-- ids : (batch_size, index + 1) 형상의 decoded sequence 행렬
-- i [scalar] : time_step 인덱스
-- cache [nested dictionary of tensors] : 고속 디코딩을 위해 keys & values에 대한 이전에 연산된 attention hidden states을 미리 저장해놓은 용도
'''
def _symbols_to_logits_fn():
    """Calculates logits of the next tokens."""
    def symbols_to_logits_fn(ids, i, temp_cache):
        # del ids
        # logits = self.decoder(ids = gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)
        # logits_t = logits[:, i, :]
        del ids
        logits = tf.cast(tf.math.log(model_fn(i)), tf.float32)
        return logits, temp_cache
    return symbols_to_logits_fn

greedy_obj = sampling_module.SamplingModule(
    length_normalization_fn=None,
    dtype=tf.float32,
    symbols_to_logits_fn=_symbols_to_logits_fn(),
    vocab_size=3,
    max_decode_length=params['max_decode_length'],
    eos_id=10,
    padded_decode=False)

ids, _ = greedy_obj.generate(initial_ids=tf.constant([9, 1]), initial_cache=cache)
print("Greedy Decoded Ids:", ids)

# %%

# inputs = masked_inputs[:, :masked_inputs.shape[1]-1]
# outputs = masked_inputs[:, :masked_inputs.shape[1]-1]

# # bos_vector = copy.deepcopy(outputs[:, :1])
# bos_vector = outputs[:, :1]
# gen_seqs = tf.cast(bos_vector, dtype = tf.int32)

# # 인코더 PAD 마스킹
# enc_pad_mask, _, _ = bart_both.mask_generator(inputs, outputs)

# # 인코더 신경망
# enc_outputs, _, _ = bart_both.encoder(inputs, enc_pad_mask, training = False)        # enc_outputs : (batch_size, seq_len, d_model)

# # 디코더 Subsequent 마스킹 (= Future 마스킹)
# _, dec_pad_mask, dec_subseq_mask = bart_both.mask_generator(inputs, gen_seqs)

# # 디코더 신경망
# dec_outputs, _ = bart_both.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

t = 0
top_k = 3
pred_probs = tf.math.top_k(dec_outputs[:, t, :], k = top_k)[0]
pred_tokens = tf.math.top_k(dec_outputs[:, t, :], k = top_k)[1]
gen_seqs_list = [[] for i in range(top_k)]
for i in range(top_k):
    gen_seqs_list[i] = tf.concat([gen_seqs, pred_tokens[:, i][:, tf.newaxis]], axis = -1)

# _, dec_pad_mask, dec_subseq_mask = bart_both.mask_generator(inputs, tiled_gen_seqs)
# dec_outputs, _ = bart_both.decoder(tiled_gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)


inputs = masked_inputs[:, :masked_inputs.shape[1]-1]
outputs = masked_inputs[:, :masked_inputs.shape[1]-1]
batch_size = inputs.shape[0]

# bos_vector = copy.deepcopy(outputs[:, :1])
bos_vector = outputs[:, :1]
gen_seqs = tf.cast(bos_vector, dtype = tf.int32)

# 인코더 PAD 마스킹
enc_pad_mask, _, _ = bart_both.mask_generator(inputs, outputs)

# 인코더 신경망
enc_outputs, _, _ = bart_both.encoder(inputs, enc_pad_mask, training = False)        # enc_outputs : (batch_size, seq_len, d_model)

max_seq_len = inputs.shape[1]
top_k = 3
tiled_gen_seqs = tf.tile(gen_seqs, multiples=[top_k, 1])            # tiled_gen_seqs : (batch_size x top_k, 1)
for t in range(max_seq_len):
    if t == 0:
        # 디코더 Subsequent 마스킹 (= Future 마스킹)
        _, dec_pad_mask, dec_subseq_mask = bart_both.mask_generator(inputs, gen_seqs)

        # 디코더 신경망
        dec_outputs, _ = bart_both.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

        '''
        beam_width 크기만큼 병렬적으로 생성과정 진행 (parallel process of generation)
        '''
        pred_logits_at_t = tf.squeeze(dec_outputs[:, -1, :])                                # pred_logits_at_t : (batch_size, vocab_size)
        top_k_pred_tokens = tf.math.top_k(pred_logits_at_t, k = top_k)[1]                   # top_k_pred_tokens : (batch_size, top_k)
        candidate_beams = tf.reshape(top_k_pred_tokens, shape = (-1, 1))                    # candidate_beams : (batch_size x top_k, 1)
        tiled_gen_seqs = tf.concat([tiled_gen_seqs, candidate_beams], axis = -1)            # tiled_gen_seqs : (batch_size x top_k, t+2)

    else:
        tensor_gen_seqs = tf.reshape(tiled_gen_seqs, shape = (-1, top_k, t+1))              # tensor_gen_seqs : (batch_size, top_k, t+1)
        tensor_gen_seqs = tf.transpose(tensor_gen_seqs, perm = (0, 2, 1))                   # tensor_gen_seqs : (batch_size, t+1, top_k)
        candidate_beams_stack = tf.zeros(shape = (0, 1), dtype = tf.int32)                  # candidate_beams_stack : (0, 1)
        candidate_beamscores_stack = tf.zeros(shape = (0, 1))                               # candidate_beamscores_stack : (0, 1)

        '''
        여기서 top_k x top_k = (top_k^2) 에 해당하는 candidate_beams가 생성됨
        '''
        for k in range(top_k):
            gen_seqs = tensor_gen_seqs[:, :, k]

            # 디코더 Subsequent 마스킹 (= Future 마스킹)
            _, dec_pad_mask, dec_subseq_mask = bart_both.mask_generator(inputs, gen_seqs)

            # 디코더 신경망
            dec_outputs, _ = bart_both.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

            '''
            beam_width 크기만큼 병렬적으로 생성과정 진행 (parallel process of generation)
            '''
            pred_logits_at_t = tf.squeeze(dec_outputs[:, -1, :])
            top_k_pred_tokens = tf.math.top_k(pred_logits_at_t, k = top_k)[1]                                              # top_k_pred_tokens : (batch_size, top_k)
            candidate_beams = tf.reshape(top_k_pred_tokens, shape = (-1, 1))                                               # candidate_beams : (batch_size x top_k, 1)
            candidate_beams_stack = tf.concat([candidate_beams_stack, candidate_beams], axis = 0)                          # candidate_beams_stack : (batch_size x (top_k^2), 1)

            '''
            beam_score 추적 및 저장 (= 각 beam 의 누적 확률)
            '''
            pred_probs_at_t = tf.squeeze(tf.nn.softmax(dec_outputs, axis = -1)[:, -1, :])
            top_k_pred_probs = tf.math.top_k(pred_probs_at_t, k = top_k)[0]                                                # top_k_pred_probs : (batch_size, top_k)
            candidate_beamscores = tf.reshape(top_k_pred_probs, shape = (-1, 1))                                           # candidate_beamscores : (batch_size x top_k, 1)
            candidate_beamscores_stack = tf.concat([candidate_beamscores_stack, candidate_beamscores], axis = 0)           # candidate_beamscores_stack : (batch_size x (top_k^2), 1)

        '''
        각 beam의 node를 확장함에 depth=2를 유지하며, width=top_k^2가 유지됨
        depthwise_candidate_beamscores를 통해 depth=2 마다 width=top_k^2의 beam이 고려되며, 거기서 top_k 만큼 다시 필터링한다.
        '''
        depthwise_candidate_beamscores = tf.reshape(candidate_beamscores_stack, shape = (batch_size, -1))
        top_k_pred_tokens_idx = tf.math.top_k(depthwise_candidate_beamscores, k = top_k)[1]
        top_k_pred_cols_idx = tf.reshape(top_k_pred_tokens_idx, shape = (-1, 1))                                                            # top_k_pred_cols_idx : (batch_size x top_k, 1)
        top_k_pred_rows_idx = tf.repeat(tf.range(batch_size), repeats = top_k)[:, tf.newaxis]                                               # top_k_pred_rows_idx : (batch_size x top_k, 1)
        top_k_pred_batch_idx = tf.concat([top_k_pred_rows_idx, top_k_pred_cols_idx], axis = -1)                                             # top_k_pred_batch_idx : (batch_size x top_k, 2)

        # # nested_candidate_beamscores_stack = tf.reshape(candidate_beamscores_stack, shape = (-1, top_k, top_k))                              # nested_candidate_beamscores_stack : (batch_size, top_k, top_k)
        # # cumprod_beamscores_stack = tf.reshape(tf.math.cumprod(nested_candidate_beamscores_stack, axis = -1), shape = (-1, top_k**2))        # cumprod_beamscores_stack : (batch_size, top_k^2)
        # # top_k_pred_tokens_idx = tf.math.top_k(cumprod_beamscores_stack, k = top_k)[1]                                                       # top_k_pred_tokens_idx : (batch_size, top_k)
        # # top_k_pred_cols_idx = tf.reshape(top_k_pred_tokens_idx, shape = (-1, 1))                                                            # top_k_pred_cols_idx : (batch_size x top_k, 1)
        # # top_k_pred_rows_idx = tf.repeat(tf.range(batch_size), repeats = top_k)[:, tf.newaxis]                                               # top_k_pred_rows_idx : (batch_size x top_k, 1)
        # # top_k_pred_batch_idx = tf.concat([top_k_pred_rows_idx, top_k_pred_cols_idx], axis = -1)                                             # top_k_pred_batch_idx : (batch_size x top_k, 2)

        # # flatten_candidate_beams_stack = tf.reshape(candidate_beams_stack, shape = (-1, top_k**2))                                           # flatten_candidate_beams_stack : (batch_size, top_k^2)

        '''
        top_k개 beams 이어 붙이기
        '''
        flatten_candidate_beams_stack = tf.reshape(candidate_beams_stack, shape = (-1, top_k**2))                                           # flatten_candidate_beams_stack : (batch_size, top_k^2)
        top_k_pred_tokens = tf.gather_nd(params = flatten_candidate_beams_stack, indices = top_k_pred_batch_idx)                            # top_k_pred_tokens (batch_size, top_k)
        candidate_beams = tf.reshape(top_k_pred_tokens, shape = (-1, 1))                                                                    # candidate_beams : (batch_size x top_k, 1)
        
        tiled_gen_seqs = tf.concat([tiled_gen_seqs, candidate_beams], axis = -1)            # tiled_gen_seqs : (batch_size x top_k, t+2)
        # tmptmp = [candidate_beams[i::3] for i in range(top_k)]
        # tmptmptmp = np.concatenate(tmptmp)
        # tiled_gen_seqs = tf.concat([tiled_gen_seqs, tmptmptmp], axis = -1)            # tiled_gen_seqs : (batch_size x top_k, t+2)

final_results = tiled_gen_seqs[0::top_k, :]

# %%
my_idx = 0
print([token_dict[t] for t in inputs[my_idx, :].numpy()])
print('\n')
print([token_dict[t] for t in final_results[my_idx, :].numpy()])




# %%
# Drug Discovery
import matplotlib.pyplot as plt
drug_reward_history1 = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/proposed/results/LEVA/reward_history_DR_epoch=1000_opt=None_lr=5e-05_lb=0_eta=0.005_es=greedy_reward=S_algo=PG_early_stop=no.csv', index_col=0)
drug_reward_history2 = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/proposed/results/LEVA/reward_history_DR_epoch=1000_opt=None_lr=5e-05_lb=0_eta=0.005_es=greedy_reward=A_algo=PG_early_stop=no.csv', index_col=0)
drug_reward_history3 = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/proposed/results/LEVA/reward_history_DR_epoch=1000_opt=None_lr=5e-05_lb=0_eta=0.005_es=greedy_reward=G_algo=PG_early_stop=no.csv', index_col=0)

plt.plot(drug_reward_history1, label = 'Sum')
plt.plot(drug_reward_history2, label = 'Average')
plt.plot(drug_reward_history3, label = 'Geometric')
plt.legend(title = 'Reward')






    # gen_molecule_image = ms[i][0]
    
    # plt.pause(0.05)
# plt.show()
# ms[1][0]
# for m in ms: Draw.MolToFile(m[0], m[1] + ".svg", size=(800, 800))


# %%
# high_BLEU -> Control_hit -> 화합물 그리기 코드

refs = [[a] for a in outputs_decoded_all]
for i in range(len(refs)):
    bleu_score = sentence_bleu(refs[i], gens_decoded_all[i], weights = [1, 0, 0, 0]) 
    if i == 0:
        bleu_score_total = [bleu_score]
    else:
        bleu_score_total += [bleu_score]

high_bleu_idx = np.argsort(bleu_score_total)[:200]

pred_label = tf.cast(tf.argmax(attr_logits, axis = -1), dtype = tf.int32)
hit_attr = attrs_all - pred_label
high_bleu_hit_attr = np.array(hit_attr)[high_bleu_idx]
high_bleu_hit_idx = np.where(high_bleu_hit_attr == 0)[0]
print(high_bleu_hit_idx)
print(bleu_score_total[high_bleu_hit_idx[0]])
gens_decoded_hit = np.array(gens_decoded_all)[high_bleu_hit_idx]
outputs_decoded_hit = np.array(outputs_decoded_all)[high_bleu_hit_idx]

from rdkit import Chem
from rdkit.Chem import Draw

# ms_smis = [["C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])Cl", "cdnb"],
#            ["C1=CC(=CC(=C1)N)C(=O)N", "3aminobenzamide"],
#            [sample_dat['Compound_smiles'][0], 'blahblah'] ]
# ms_smis = [[ms_smis[i][0].upper(), 'blah'] for i in range(len(ms_smis))]

# ms_smis = list(outputs_decoded_hit)[:3]
ms_smis = list(gens_decoded_hit)
ms_smis = [[ms_smis[i].upper(), 'blah'] for i in range(len(ms_smis))]

ms = [[Chem.MolFromSmiles(x[0]), x[1]] for x in ms_smis]
for i in range(len(ms)):
    display(ms[i][0])

# print(bleu_score_total)