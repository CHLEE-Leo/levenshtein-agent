# %%
'''
model.py file provides any instances related to main model, not baselines.
'''
from operator import concat
import sys
from attr import attrib
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import copy
from utils import *
from tqdm import tqdm

# (1) 포지션 인코더
class Position_Encoder:
    def __init__(self, **kwargs):
        self.d_model = kwargs['d_model']

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, len_pos, d_model):
        positions = np.arange(len_pos)              # position list array 만들기
        dimension_indices = np.arange(d_model)      # dimension index list array 만들기
        angle_rads =  self.get_angles(positions[:, np.newaxis], 
                                        dimension_indices[np.newaxis, :], 
                                        d_model)

        # 어레이의 짝수 인덱스에 sin을 적용 (2i)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 어레이의 홀수 인덱스에 cos를 적용; (2i+1)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # pos_encoding은 (1, len_pos, d_model) 크기의 차원을 가짐
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __call__(self, embedding_matrix):
        len_pos = embedding_matrix.shape[1]
        pos_encoding = self.positional_encoding(len_pos, self.d_model)

        return pos_encoding

# (2) 마스크 제너레이터
class Mask_Generator:
    '''
    padding_mask : [PAD] 토큰 (= 0번 토큰)만 마스킹하는 함수
    subsequent_mask : 일명 'look_ahead_mask' 함수로, 미래의 타임스텝을 모두 마스킹하는 함수
    '''
    def padding_mask(self, seq):
        # keras.tokenizer를 사용할경우, padding token 값은 0이되므로, 토큰값이 0인 위치만 찾아서 True (1), 나머지 토큰값들은 False (0)을 만들어준다.
        seq = tf.cast(tf.math.equal(seq, 0), dtype = tf.float32)

        # attention_map = attention_logits 의 차원은 4차원 ((batch_size, num_heads, seq_len, seq_len))이므로 
        # padding_mask도 4차원으로 만들어주어야 함.
        padding_mask = seq[:, np.newaxis, np.newaxis, :] * -10e9

        return padding_mask

    # (대칭행렬인) attention_map에 대하여, 현재 position 이후의 token들에 대해서 mask를 씌워
    # 참조를 못하게 하는함수
    def subsequent_mask(self, tar_len):
        '''
        tf.linalg.band_part(M, n, m) : 정방행렬 M의 대각원소를 기준으로 하삼각행렬의 n번째 띠, 상삼각행렬의 m번째 띠까지 행렬 M의 값을 복원 (n, m보다 큰 값의 띠는 0으로 마스킹)
        '''
        # 상삼각함수만 1로 만들기
        mask_matrix = 1 - tf.linalg.band_part(tf.ones((tar_len, tar_len)), -1, 0)
        subseq_mask = mask_matrix * -10e9       # subseq_mask : (seq_len, seq_len)
        return subseq_mask

    def window_mask(self, tar_len, window_len):                
        window_buffer_size = window_len // 2
        mask_matrix = 1 - tf.linalg.band_part(tf.ones(shape = (tar_len, tar_len)), window_buffer_size, window_buffer_size)
        window_mask = mask_matrix * -10e9
        return window_mask

    def __call__(self, inp, tar):
        # 인코더에서 패딩 부분 마스크
        enc_padding_mask = self.padding_mask(inp)
        # enc_padding_mask = self.masking(inp, [0, 1, 3, 4])

        # 디코더에서 사용되는 인코더의 아웃풋에서 패딩 부분 마스크 (디코더의 두번째 어텐션 블록에서 활용)
        dec_padding_mask = self.padding_mask(inp)
        # dec_padding_mask = self.masking(inp, [0, 1, 3, 4])
        
        # subsequent mask를 생성하기 위해 target의 length 정의
        tar_len = tar.shape[1]
        dec_subsequent_mask = self.subsequent_mask(tar_len)

        return enc_padding_mask, dec_padding_mask, dec_subsequent_mask    

# 멀티 헤드 어텐션
class MultiHeadAttention(tf.keras.layers.Layer):        
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self).__init__()

        # 하이퍼 파라미터 정의
        self.d_model = kwargs['d_model']
        self.num_heads = kwargs['num_heads']
        self.depth = self.d_model // self.num_heads

        # linear projection 함수 : embedding to attnetion 
        self.wq = tf.keras.layers.Dense(units = self.d_model)
        self.wk = tf.keras.layers.Dense(units = self.d_model)
        self.wv = tf.keras.layers.Dense(units = self.d_model)

        # linear proejection 함수 : scaled attention to output
        self.linear_projection = tf.keras.layers.Dense(units = self.d_model)


    # (linearly-projected) embedding_vector를 head갯수로 나누는 함수
    def split_heads(self, x, batch_size):
        
        # d_model을 head 갯수로 나눠 depth크기를 만들기 위해서 나머지 모든 차원들은 특정 값으로 고정되어야 함.
        # 텐서의 축은 (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, depth)가 되어야 함
        # 이를 위해서 batch_size라는 파라미터를 사전에 설정하여 고정해줄 필요가 있음.

        x = tf.reshape(x, shape = (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])        

    def scaled_dot_product_attention(self, q, k, v, mask):
        
        # print('q :', q.shape)
        # print('k :', k.shape)
        # print('v :', v.shape)

        # attention_map의 차원 : (batch_size, num_heads, seq_len1, seq_len2)
        attention_map = tf.matmul(q, k, transpose_b = True)

        # scale 적용
        # attention_logits : (batch_size, num_heads, seq_len1, seq_len2)
        dk = tf.cast(k.shape[-1], dtype = tf.float32)
        attention_logits = attention_map / tf.math.sqrt(dk)

        # masking 적용
        # pad_mask : (batch_size, 1, 1, seq_len)
        # subsequent_mask : (seq_len, seq_len)
        if mask is not None:
            if len(mask.shape) == 4:
                attention_logits += mask[:, :, :, :tf.shape(attention_logits)[-1]]
            else:
                attention_logits += mask

        # attention_weights의 차원 : (batch_size, num_heads, seq_len1, seq_len2)
        attention_weights = tf.nn.softmax(attention_logits, axis = -1)

        # attention_scores의 차원 : (batch_size, num_heads, sequnece_len1, depth)
        attention_scores = tf.matmul(attention_weights, v)

        return attention_scores, attention_weights

    def call(self, query, key, value, mask):
        # query, key, value는 embedded representation된 문장 데이터들이다.
        # mask는 미리 뽑아서 여기까지 계속 전달해주어야 함.

        # 임베딩 벡터를 linear projection 해주기
        # Q, K, V의 차원 : (batch_size, seq_len, d_model)
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        # linearly projected된 임베딩 벡터를 멀티헤드로 쪼개주기
        # q, k, v는 (batch_size, num_heads, seq_len, depth) 4차원임.
        q = self.split_heads(Q, tf.shape(Q)[0])
        k = self.split_heads(K, tf.shape(K)[0])
        v = self.split_heads(V, tf.shape(V)[0])

        # scaled_dot_product_attention 적용해주기
        # scaled_attention_scores : (batch_size, num_heads, seq_len1, depth)
        scaled_attention_scores, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # self-attention output의 multi head들을 concat해주기
        scaled_attention_scores = tf.transpose(scaled_attention_scores, perm = [0, 2, 1, 3])
        scaled_attention_scores = tf.reshape(scaled_attention_scores, shape = (tf.shape(Q)[0], -1, self.d_model))   # scaled_attention_scores : (batch_size, seq_len1, d_model)

        # concat된 output에 linear-proejction 적용
        output = self.linear_projection(scaled_attention_scores)    # output : (batch_size, seq_len1, d_model)

        return output, attention_weights

# 포지션-와이즈 피드포워드 네트워크
class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeedForwardNetwork, self).__init__()
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']

        # linear proejection 함수 : normalized scaled attention to output
        self.linear_projection1 = tf.keras.layers.Dense(units = self.d_ff, activation = 'relu')
        self.linear_projection2 = tf.keras.layers.Dense(units = self.d_model)

    def call(self, x):
        # 여기서 x는 add & norm 레이어를 통과한 값
        output = self.linear_projection1(x)
        output = self.linear_projection2(output)

        return output

# 인코더 레이어
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(**kwargs)
        self.ffn = FeedForwardNetwork(**kwargs)
        
        self.dropout1 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.normalization1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

        self.dropout2 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.normalization2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

    def call(self, x, mask):
        mha_outputs, att_weights = self.mha(x, x, x, mask)
        mha_outputs = self.dropout1(mha_outputs)
        out1 = self.normalization1(mha_outputs + x)                 # out1 : (batch_size, seq_len, d_model)

        ffn_outputs = self.ffn(out1)
        ffn_outputs = self.dropout2(ffn_outputs)
        out2 = self.normalization2(ffn_outputs + out1)              # out2 : (batch_size, seq_len, d_model)

        return out2, att_weights

# 디코더 레이어
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DecoderLayer, self).__init__()
        self.batch_size = kwargs['batch_size']

        self.mha1 = MultiHeadAttention(**kwargs)
        self.mha2 = MultiHeadAttention(**kwargs)
        self.ffn = FeedForwardNetwork(**kwargs)

        self.dropout1 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.dropout2 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.dropout3 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])

        self.normalization1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.normalization2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.normalization3 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
    

    def call(self, x, enc_outputs, dec_pad_mask, dec_subseq_mask):
        '''
        x : (batch_size, seq_len, d_model)
        enc_outputs : (batch_size, seq_len, d_model)
        '''

        # masked self-attention
        mha_outputs1, attn_weights1 = self.mha1(x, x, x, dec_subseq_mask)    # q, k, v, mask; q, k, v의 차원 : (batch_size, seq_len, d_model)
        mha_outputs1 = self.dropout1(mha_outputs1)
        out1 = self.normalization1(mha_outputs1 + x)        # out1 : (batch_size, seq_len, d_model)

        # cross-attention
        mha_outputs2, attn_weights2 = self.mha2(out1, enc_outputs, enc_outputs, dec_pad_mask)
        mha_outputs2 = self.dropout2(mha_outputs2)
        out2 = self.normalization2(mha_outputs2 + out1)     # out2 : (batch_size, seq_len, d_model)

        # feed_forward
        ffn_outputs = self.ffn(out2)
        ffn_outputs = self.dropout3(ffn_outputs)
        out3 = self.normalization3(ffn_outputs + out2)      # out3 : (bach_size, seq_len, d_model)

        return out3, attn_weights1, attn_weights2

# 인코더 모듈
class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        # 하이퍼 파라미터
        self.d_model = kwargs['d_model_enc']
        self.enc_dict_len = kwargs['vocab_size']
        self.num_layers_enc = kwargs['num_layers_enc']
        self.position_encoder = Position_Encoder(**kwargs)

        # 임베딩 레이어
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.enc_dict_len, output_dim = self.d_model, mask_zero = False)        

        # 같은걸 여러번 쌓기 (For Normalizing Flow Effect)
        if kwargs['stack'] == 'rnn':
            self.encoder_layer = EncoderLayer(**kwargs)
            self.stacked_enc_layers = [self.encoder_layer for i in range(self.num_layers_enc)]

        # 다르게 여러번 쌓기 (For general Case)
        elif kwargs['stack'] == 'mlp':
            self.stacked_enc_layers = [EncoderLayer(**kwargs) for i in range(self.num_layers_enc)]

    def call(self, inputs, enc_pad_mask):

        # 임베딩 레이어
        inputs_embeddings = self.embedding_layer(inputs)
        inputs_embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype = tf.float32))
        inputs_embeddings += self.position_encoder(inputs_embeddings)

        # 인코더 레이어    
        z = inputs_embeddings
        for enc_layer in self.stacked_enc_layers:
            z, att_weights = enc_layer(z, enc_pad_mask)

        # 출력 레이어
        enc_outputs = z     # (batch_size, seq_len, d_model)
        return enc_outputs, inputs_embeddings, att_weights

# 디코더 모듈
class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        # 하이퍼 파라미터
        self.d_model = kwargs['d_model_dec']
        self.dec_dict_len = kwargs['vocab_size']
        self.num_layers = kwargs['num_layers_dec']
        self.position_encoder = Position_Encoder(**kwargs)

        # 임베딩 레이어
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.dec_dict_len, output_dim = self.d_model, mask_zero = False)        

        # 같은걸 여러번 쌓기 (For Normalizing Flow Effect)
        if kwargs['stack'] == 'rnn':
            self.decoder_layer = DecoderLayer(**kwargs)
            self.stacked_dec_layers = [self.decoder_layer for i in range(self.num_layers)]

        # 다르게 여러번 쌓기 (For general Case)
        elif kwargs['stack'] == 'mlp':
            self.stacked_dec_layers = [DecoderLayer(**kwargs) for i in range(self.num_layers)]
        
        # 출력 레이어
        self.linear_layer = tf.keras.layers.Dense(units = self.dec_dict_len)

    def call(self, outputs, enc_outputs, dec_pad_mask, dec_seq_mask):

        attn_weights_dict = {}

        # 임베딩 레이어
        outputs = self.embedding_layer(outputs)  
        outputs *= tf.math.sqrt(tf.cast(self.d_model, dtype = tf.float32))
        outputs += self.position_encoder(outputs)

        # 디코더 레이어
        for i, dec_layer in enumerate(self.stacked_dec_layers):
            outputs, attn_w1, attn_w2 = dec_layer(outputs, enc_outputs, dec_pad_mask, dec_seq_mask)                          # outputs : (batch_size, seq_len, d_model)
            attn_weights_dict['decoder_layer{}_attn_block1'.format(i+1)] = attn_w1
            attn_weights_dict['decoder_layer{}_attn_block2'.format(i+1)] = attn_w2

        # 출력 레이어
        final_outputs = self.linear_layer(outputs)    # final_outputs : (batch_size, seq_len, voca_size)

        return final_outputs, attn_weights_dict

# AETransformer
class AETransformer(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AETransformer, self).__init__()

        # 모델 관련 공유 파라미터
        self.batch_size = kwargs['batch_size']
        self.mask_generator = Mask_Generator()
        # self.initial_embedding_weights = None

        # 인코더 파라미터
        kwargs['d_model'] = kwargs['d_model_enc']
        kwargs['num_heads'] = kwargs['num_heads_enc']
        self.encoder = Encoder(**kwargs)
        # self.encoder.embedding_layer.weights = None

        # 디코더 파라미터
        kwargs['d_model'] = kwargs['d_model_dec']
        kwargs['num_heads'] = kwargs['num_heads_dec']
        self.decoder = Decoder(**kwargs)
        # self.decoder.embedding_layer.weights = None

    def call(self, data):
        # 인풋 데이터 및 패딩토큰 마스킹 행렬 준비
        inputs, outputs = data
        enc_pad_mask, _, _ = self.mask_generator(inputs, inputs)
        _, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, outputs)

        # Encoder 네트워크
        enc_outputs, inputs_embeddings, att_weights = self.encoder(inputs, enc_pad_mask)         # enc_outputs : (batch_size, seq_len, d_model)

        # Decoder 네트워크
        dec_outputs, _ = self.decoder(outputs, enc_outputs, dec_pad_mask, dec_subseq_mask)      # decoder_outputs : (batch_size, seq_len, voca_size)

        return enc_outputs, inputs_embeddings, dec_outputs, att_weights

    def token_sampling(self, dec_outputs, top_k, strategy = 'greedy'):

        # 탐욕 샘플링
        if strategy == 'greedy':
            top_k = 1
            preds = tf.math.top_k(dec_outputs[:, -1, :], k = top_k)[1]          # get a token of maximal probability at each time step
            # preds = tf.expand_dims(preds, axis = -1)                            # get a token of the last time step
            preds = tf.cast(preds, dtype = tf.int32)                            # set a tensor type as float32

        # 확률 샘플링
        elif strategy == 'stochastic':
            preds = tf.random.categorical(logits = dec_outputs[:, -1, :], num_samples = 1)
            preds = tf.cast(preds, dtype = tf.int32)            # set a tensor type as float32

        # top-k 샘플링
        elif strategy == 'top_k':
            top_k_prob_vec = tf.math.top_k(dec_outputs[:, -1, :], k = top_k)[0]     # get the top-k probability vector (*** 사실 top-k logits임)
            top_k_prob_vec_redist = tf.nn.softmax(top_k_prob_vec, axis = -1)        # redistrbute the probability mass among top-k tokens
            top_k_token_vec = tf.math.top_k(dec_outputs[:, -1, :], k = top_k)[1]    # get the token vector of top-k probability

            col_idx_vec = tf.cast(tf.random.categorical(logits = top_k_prob_vec_redist, num_samples = 1), dtype = tf.int32)                    # sample the idx of logit vector according to logit value, and set it as col_idx vector
            batch_size = tf.shape(top_k_prob_vec_redist)[0]                                    # get the batch_size
            row_idx_vec = tf.reshape(tf.range(batch_size), shape = (-1, 1))             # craete idx vector of batch_size and set it as the row_idx vector

            selected_idx = tf.concat([row_idx_vec,  col_idx_vec], axis = -1)            # concat row_idx vector and col_idx vector by column axis

            preds = tf.reshape(tf.gather_nd(top_k_token_vec, selected_idx), shape = (-1, 1))        # obtain the values from the token vector of top-k probability, which is the sampled token vector.

        # 빔서치 샘플링
        elif strategy == 'beam':
            batch_size = tf.shape(dec_outputs)[0]
            seq_len = tf.shape(dec_outputs)[1]
            seq_len_vec = tf.repeat(seq_len, repeats = batch_size)
            
            dec_outputs_t = tf.transpose(dec_outputs, perm = [1, 0, 2])
            preds, probs = tf.nn.ctc_beam_search_decoder(dec_outputs_t, seq_len_vec, beam_width = beam_width, top_paths = top_paths)

        # # 뉴클리우스 샘플링
        # elif strategy == 'top_p':   # a.k.a nucleus sampling
        #     s

        return preds

    # 추론 단계 함수 (생성 과정) - Quantile Token 없는 경우
    def inference(self, data, prompt_vector, token_dict, decoding = 'greedy', top_k = 3):
        inputs, outputs = data

        '''
        마스크 시퀀스의 앞에 프롬프트 및 [BOS] 토큰 붙여주기
        '''
        # bos_vector = copy.deepcopy(outputs[:, :1])
        bos_vector = outputs[:, :1]
        gen_seqs = tf.cast(tf.concat([prompt_vector, bos_vector], axis = -1), dtype = tf.int32)

        # 인코더 PAD 마스킹
        enc_pad_mask, _, _ = self.mask_generator(inputs, outputs)

        # 인코더 신경망
        enc_outputs, _, _ = self.encoder(inputs, enc_pad_mask, training = False)        # enc_outputs : (batch_size, seq_len, d_model)

        # for t in range(0, inputs.shape[1] - 1):

        #     # 디코더 Subsequent 마스킹 (= Future 마스킹)
        #     _, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, gen_seqs)

        #     # 디코더 신경망
        #     dec_outputs, _ = self.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

        #     # 시퀀스 생성
        #     preds = self.token_sampling(dec_outputs, top_k = top_k, strategy = decoding)            # get predicted tokens
        #     gen_seqs = tf.concat([gen_seqs, preds], axis = -1)                                      # concatenate the token of last step to the gen_seq

        done = True
        t = 0
        while done:

            # ['eos'] 토큰이 적어도 한번 생성된 시퀀스의 갯수 확보
            first_eos_idx = slicing_eos_token_index(gen_seqs, token_dict)
            num_eos_seq = tf.shape(first_eos_idx)[0]

            # 현 시점 t가 인풋 시퀀스의 총 길이보다 작을 경우
            if t < tf.shape(inputs)[1]-1:
                # 배치 내 모든 생성 시퀀스에 ['eos'] 토큰이 등장하지 않은 경우, while문 지속
                if num_eos_seq < tf.shape(inputs)[0]:

                    t += 1

                    # 디코더 Subsequent 마스킹 (= Future 마스킹)
                    _, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, gen_seqs)

                    # 디코더 신경망
                    dec_outputs, _ = self.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

                    # 시퀀스 생성
                    preds = self.token_sampling(dec_outputs, top_k = top_k, strategy = decoding)            # get predicted tokens
                    gen_seqs = tf.concat([gen_seqs, preds], axis = -1)                                      # concatenate the token of last step to the gen_seq

                # 배치 내 모든 생성 시퀀스에 ['eos'] 토큰이 등장한 경우, 나머지 모든 토큰에 ['pad']를 붙이고 while문 종료
                else:
                    rest_seq_len = tf.shape(inputs)[1] - 1 - t
                    rest_pad_mat = tf.ones(shape = (tf.shape(inputs)[0], rest_seq_len), dtype = tf.int32) * tf.cast(get_token(token_dict, '[pad]'), dtype = tf.int32)
                    gen_seqs = tf.concat([gen_seqs, rest_pad_mat], axis = -1)

                    # _, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, gen_seqs)
                    # dec_outputs, _ = self.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

                    done = False                

            # 현 시점 t가 인풋 시퀀스의 총 길이와 같거나 그보다 클 경우
            else:
                rest_seq_len = tf.shape(inputs)[1] - 1 - t
                rest_pad_mat = tf.ones(shape = (tf.shape(inputs)[0], rest_seq_len), dtype = tf.int32) * tf.cast(get_token(token_dict, '[pad]'), dtype = tf.int32)
                gen_seqs = tf.concat([gen_seqs, rest_pad_mat], axis = -1)

                # _, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, gen_seqs)
                # dec_outputs, _ = self.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

                done = False

        return gen_seqs

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
RL 에이전트
'''
class LEVAgent(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LEVAgent, self).__init__()

        # 파라미터
        self.batch_size = kwargs['batch_size']
        self.mask_generator = Mask_Generator()
        kwargs['d_model'] = kwargs['d_model_enc']
        kwargs['num_heads'] = kwargs['num_heads_enc']
        self.action_size = len(kwargs['action_space'])

        # 인코더
        self.encoder = Encoder(**kwargs)
        self.encoder.position_encoder = Position_Encoder(**kwargs)

        # 출력 레이어
        self.linear_layer = tf.keras.layers.Dense(units = self.action_size)

    def call(self, inputs, enc_pad_mask):

        # 인코더
        enc_outputs, _, att_weights = self.encoder(inputs, enc_pad_mask)         # enc_outputs : (batch_size, seq_len, d_model)

        # 출력 레이어
        agent_outputs = self.linear_layer(enc_outputs)

        return agent_outputs, att_weights

class Reward_Function(tf.keras.Model):
    def __init__(self, num_attributes, **kwargs):
        super(Reward_Function, self).__init__()

        # 파라미터
        self.batch_size = kwargs['batch_size']
        self.mask_generator = Mask_Generator()
        kwargs['d_model'] = kwargs['d_model_enc']
        kwargs['num_heads'] = kwargs['num_heads_enc']

        # 인코더
        self.encoder = Encoder(**kwargs)
        self.encoder.position_encoder = Position_Encoder(**kwargs)

        # 선형 레이어
        self.linear = tf.keras.layers.Dense(units = num_attributes)

    def call(self, inputs, mask):
        '''
        inputs : 시퀀스 샘플
        '''

        # 인코더
        enc_outputs, _, _ = self.encoder(inputs, mask)         # enc_outputs : (batch_size, seq_len, d_model)

        # 선형 레이어
        pred_reward = self.linear(tf.reduce_sum(enc_outputs, axis = 1))      # pred_reward : (batch_size, 2)

        return pred_reward