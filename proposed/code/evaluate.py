# %%
import argparse
import os
from sys import getsizeof
from tkinter import E
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle, json
import copy
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from model import *
from utils import *
from main import get_params
import gc
import time
import copy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--task', type = str, required = True)      # {ST, DD}
parser.add_argument('--model', type = str, required = True)     # {proposed, baseline}
parser.add_argument('--metric', type = str, required = True)    # {ACC, BLEU, PPL}]
args = parser.parse_args()
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# ################################################################################################################################################
'''
(control) Accuracy : ACC
(content) Perservation : BLEU_1, BLEU_2, BLEU_3, BLEU_4, GLEU
Naturalness : PPL
'''
# ################################################################################################################################################
if args.task == 'ST':

    # 실험 데이터 로드
    test_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/input_sequence(test).npy')
    eos_idx = indexing_eos_token(test_input_sequence)
    test_input_sequence = test_input_sequence[np.where(eos_idx >= 4)[0], :]             # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
    test_attribute = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/attribute(test).npy')
    test_attribute = test_attribute[np.where(eos_idx >= 4)[0]]                            # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

    # 토큰 사전 가져오기
    with open('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer' + '/token_dict.pickle', 'rb') as f:
        token_dict = pickle.load(f)
    special_token_list = ['[pad]', '[mask]']
    reward_class_token_list = ['[' + 'R_' + str(reward_class) + ']' for reward_class in range(len(np.unique(test_attribute)))]
    edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
    add_token_list = special_token_list + reward_class_token_list + edit_token_list
    token_dict = add_token_list_in_dict(add_token_list, token_dict)
    action_set = list(token_dict.values())[-len(edit_token_list):]

    # "리버스" 프롬프트 벡터 (= 보상 클래스 토큰) 생성 준비
    test_reward_class_vector = np.ones(shape = test_attribute.shape).astype(np.int32)
    for reward_class in range(len(np.unique(test_attribute))):
        rev_reward_class = np.unique(test_attribute)[np.where(reward_class != np.unique(test_attribute))[0]][0]
        test_reward_class_vector[np.where(test_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(rev_reward_class) + ']')

    # 프롬프트 벡터 생성
    test_prompt_vector = test_reward_class_vector[:, np.newaxis]

    # BLEU 계산용 refereces 데이터셋 만들기
    references = get_decoded_list(test_input_sequence, token_dict)


    if args.model == 'proposed':
        target_task = args.task          # {ST, DR}
        decoding_mode = input('decoding= ')     # {AR, NAR}
        num_epochs = input('num_epochs=')
        batch_size = input('batch_size=')
        lr = input('lr=')
        opt_type = input('opt=')                # {None, CosDecay, CosDecayRe, ExpDecay, InvTimeDecay}
        len_buffer = input('len_buffer(lb)= ')
        eta = input('eta=')
        es = input('es=')                       # {greedy, stochastic}
        reward_type = input('reward=')          # {S, A, G, H}
        algo_type = input('algo=')              # {None, PG, PPO}
        early_stop = input('early_stop=')

        # Dataset 객체 자체는 cpu에 할당
        with tf.device("/cpu:0"):
            dataset = tf.data.Dataset.from_tensor_slices((test_input_sequence, test_prompt_vector, test_attribute))
            batchset = dataset.batch(batch_size = test_input_sequence.shape[0], num_parallel_calls = 8)
            batchset = batchset.prefetch(1)

        # ---------------------------------------------------------------------------------------- #
        # 2-1) 에이전트 모델 초기화
        # leva_kwargs_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/LEVA/kwargs_LEVA_ST_500'
        leva_kwargs_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/LEVA/kwargs_LEVA' + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)

        with open(leva_kwargs_dir, 'r') as f:
            LEVA_kwargs = json.load(f)
        lev_agent = LEVAgent(**LEVA_kwargs)

        # --> 초기 정책모델 학습 가중치 로드하여 타겟 정책모델 가중치 초기화
        # load_dir2 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/LEVA_coBART_ST_rnn_500' + str('_lb=' + len_buffer)
        # load_dir2 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/LEVA_ST_500_lb=0_eta=0.005_es=greedy_reward=G'
        if algo_type != "None":
            load_dir2 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/LEVA_ST' + str('_epoch=' + num_epochs + '_opt=' + opt_type + '_lr=' + lr + '_lb=' + len_buffer + '_eta=' + eta + '_es=' + es + '_reward=' + reward_type + '_algo=' + algo_type + '_early_stop=' + early_stop)
        else:
            load_dir2 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/LEVA_ST' + str('_epoch=' + num_epochs + '_opt=' + opt_type + '_lr=' + lr + '_lb=' + len_buffer + '_eta=' + eta + '_es=' + es + '_reward=' + reward_type)

        print('lev_agent_weights_dir : ' , load_dir2)
        lev_agent.load_weights(tf.train.latest_checkpoint(load_dir2))

        # 2-2) 환경 모델 초기화
        env_gen_model_name = 'NAR'
        # env_gen_model_name = 'coBART'

        prefix = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/' + env_gen_model_name
        env_gen_kwargs = '/kwargs_' + env_gen_model_name + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
        env_gen_kwargs_dir = prefix + env_gen_kwargs
        with open(env_gen_kwargs_dir, 'r') as f:
            env_kwargs = json.load(f)

        env_gen_model = AETransformer(**env_kwargs)

        # --> 초기 환경 모델 학습 가중치 로드하여 타겟 정책모델 가중치 초기화
        prefix = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/'
        env_gen_weights_dir = prefix + env_gen_model_name + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
        print(env_gen_weights_dir)
        env_gen_model.load_weights(tf.train.latest_checkpoint(env_gen_weights_dir))
        # print('env_gen_weights_dir : /home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/coBART_ST_500')
        # env_gen_model.load_weights(tf.train.latest_checkpoint('/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/coBART_ST_500'))

        # ---------------------------------------------------------------------------------------- #
        # 5) 훈련 루프

        # --> 메트릭 초기화
        metrics_names = ['lev_loss', 'div_score', 'reward']
        pre_reward = 0
        reward_history = []

        # --> 루프 수행
        for idx, (inputs, attr_tokens, attrs) in enumerate(batchset):        

            enc_pad_mask_for_agent, _, _ = env_gen_model.mask_generator(inputs, inputs)

            '''
            편집 계획 (edit_plans) 생성
            '''
            agent_outputs, action_att_weights = lev_agent(inputs, enc_pad_mask_for_agent, training = False)
            agent_actions, action_tokens = return_action_token(agent_outputs, inputs, token_dict, mode = 'test')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                    # action_tokens = action의 토큰 : (batch_size, seq_len)
            
            '''
            레반슈타인 연산 수행
            '''
            # 레반슈타인 연산자 적용
            masked_inputs, new_action_tokens = apply_lev_operation(inputs, action_tokens, token_dict)
            masked_inputs = masked_inputs.numpy()                       # numpy indexing을 위해 자료형을 텐서에서 넘파이로 바꿔주기
            masked_inputs_copy = copy.deepcopy(masked_inputs)           # masked_inputs는 아래에서 inference 과정동안 계속 update 될 예정이므로 masked_inputs_copy를 따로 만들어 놓기.

            # AutoRegressive 디코딩
            if decoding_mode == 'AR':

                '''
                스타일 전환 : attribute 코드 prefix하기
                '''
                bos_vector = masked_inputs[:, :1]    # [bos] 토큰 벡터
                gen_seqs = np.concatenate([attr_tokens, bos_vector], axis = -1)

                for t in range(masked_inputs.shape[1]-2):

                    # t가 0일 때
                    if t == 0:
                        # 인코더 PAD 마스킹
                        enc_pad_mask_for_env, _, _ = env_gen_model.mask_generator(masked_inputs, masked_inputs)

                        # 인코더 신경망
                        enc_outputs, _, _ = env_gen_model.encoder(masked_inputs, enc_pad_mask_for_env, training = False)        # enc_outputs : (batch_size, seq_len, d_model)

                    # t가 0보다 클 때
                    else:
                        # masked_inputs 업데이트
                        masked_inputs[:, :gen_seqs.shape[1]-1] = gen_seqs[:, 1:]    # gen_seqs에서 첫번째 토큰인 prompt 토큰은 제외해야 하므로 gen_seqs[:, 1:]로 셋팅= ['bos'] 부터 시작.
                                                                                    # gen_seqs에서 첫번째 토큰을 제외했으므로, masked_inputs에 삽입되는 gen_seqs의 shape를 정의해줄 떄 shape[1]-1 해준다.

                        # 인코더 PAD 마스킹
                        enc_pad_mask_for_env, _, _ = env_gen_model.mask_generator(masked_inputs, masked_inputs)

                        # 인코더 신경망
                        enc_outputs, _, _ = env_gen_model.encoder(masked_inputs, enc_pad_mask_for_env, training = False)        # enc_outputs : (batch_size, seq_len, d_model)


                    # 디코더 Subsequent 마스킹 (= Future 마스킹)
                    _, dec_pad_mask, dec_subseq_mask = env_gen_model.mask_generator(masked_inputs, gen_seqs)

                    # 디코더 신경망
                    dec_outputs, _ = env_gen_model.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

                    # 토큰 예측
                    pred_tokens = tf.cast(tf.math.argmax(dec_outputs[:, -1, :], axis = -1), dtype = tf.int32)[:, tf.newaxis]

                    # 시퀀스 생성
                    gen_seqs = tf.concat([gen_seqs, pred_tokens], axis = -1)
                    gen_seqs = gen_seqs.numpy()


            # Non-AutoRegressive 디코딩
            elif decoding_mode == 'NAR':

                '''
                스타일 전환 : attribute 코드를 prefix하여 환경 모델에 입력
                '''
                # 마스크 인풋 시퀀스의 앞에 프롬프트 붙여주기
                masked_prompt_inputs = np.concatenate([attr_tokens, masked_inputs], axis = -1)

                # 토큰 예측
                _ ,_, dec_outputs, _ = env_gen_model((masked_inputs, masked_prompt_inputs[:, :masked_prompt_inputs.shape[1]-1]), training = False)

                # 시퀀스 생성
                gen_seqs = tf.cast(tf.math.argmax(dec_outputs, axis = -1), dtype = tf.int32)
                gen_seqs = gen_seqs.numpy()


            '''
            제어된 시퀀스 생성
            '''
            controlled_gens = fill_pad_after_eos(gen_seqs, masked_inputs_copy, token_dict)

        del env_gen_model, lev_agent    

        '''
        my_idx : 0-499 --> neg
        my_idx : 500-999 --> pos
        '''
        for my_idx in range(10, 20):
            print('attrs : {}'.format(attrs[my_idx]))
            print('inputs : {}'.format([token_dict[t] for t in inputs[my_idx, :].numpy()]))
            print('edit plans : {}'.format([token_dict[t] for t in action_tokens[my_idx, :].numpy()]))
            print('gens : {}'.format([token_dict[t] for t in controlled_gens[my_idx, :].numpy()]))
            print('\n')

        gens = controlled_gens[:, 1:]
        gens_decoded = get_decoded_list(gens, token_dict)
        gens_decoded_string = list(map(' '.join, gens_decoded))
        gens_decoded_string = [i.replace('. )', '.') for i in gens_decoded_string]

        with open('/home/messy92/Leo/NAS_folder/ICML23/proposed/results/LEVA/neg_to_pos.txt', 'w') as file:
            file.writelines(each_gen + '\n' for each_gen in gens_decoded_string[:500])
        with open('/home/messy92/Leo/NAS_folder/ICML23/proposed/results/LEVA/pos_to_neg.txt', 'w') as file:
            file.writelines(each_gen + '\n' for each_gen in gens_decoded_string[500:])

        if args.metric == 'ACC':
            m = tf.keras.metrics.CategoricalAccuracy()
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
            # '''
            # If you first load distil_BERT from huggingface ropo, save the weights at local directory.
            # '''
            # # checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
            # checkpoint = "ydshieh/bert-base-uncased-yelp-polarity"
            # hf_bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            # distil_BERT = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
            # # MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/Distil_BERT'
            # MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/yelp_Base_BERT'
            # if os.path.exists(MODEL_SAVE_PATH):
            #     print(f"{MODEL_SAVE_PATH} -- Folder already exists \n")
            # else:
            #     os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            #     print(f"{MODEL_SAVE_PATH} -- Folder create complete \n")
            # # save tokenizer, model
            # distil_BERT.save_pretrained(MODEL_SAVE_PATH)
            # hf_bert_tokenizer.save_pretrained(MODEL_SAVE_PATH)
            '''
            if you already have weights for distil_BERT at your local directory, then just load it.
            '''
            # Load Fine-tuning model
            # bert_lr = input('bert_lr=')
            # distil_bert_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/Distil_BERT_epoch=10_batch_size=512' + str('_lr=' + bert_lr)
            # hf_bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            distil_bert_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/yelp_Base_BERT'
            hf_bert_tokenizer = AutoTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
            hf_bert_tokenizer.pad_token = '[pad]'
            distil_BERT = TFAutoModelForSequenceClassification.from_pretrained(distil_bert_dir)

            # 허깅페이스 distilbert 기반 분류기 활용
            # gens_decoded_string = [" ".join(a_gen) for a_gen in gens_decoded]
            gens_encoded = hf_bert_tokenizer(gens_decoded_string, padding=True, truncation=True, return_tensors="tf")
            pred_outputs = distil_BERT(gens_encoded, training = False)
            # pred_probs = tf.nn.softmax(pred_outputs.logits, axis = -1)
            rev_attr_onehot = tf.one_hot(1 - test_attribute, depth = len(tf.unique(test_attribute)[0]))
            # mean_score = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(pred_probs, rev_attr_onehot), axis = -1))
            m.update_state(rev_attr_onehot, pred_outputs.logits)

            # print('model_name : {},\n ACC : {}'.format(args.model, mean_score.numpy()))
            print('model_name : {},\n ACC : {}'.format(args.model, m.result().numpy()))

        elif args.metric == 'BLEU':
            from nltk.translate.bleu_score import corpus_bleu
            from nltk.translate.gleu_score import corpus_gleu

            refs = [[a] for a in references]
            bleu_percent_list = []
            for i in range(4):
                n_gram_setup = [0, 0, 0, 0]
                n_gram_setup[i] = 1
                bleu_score = corpus_bleu(refs, gens_decoded, weights = n_gram_setup)    # weights : n-gram 조절 (BLEU_1, BLEU_2, BLEU_3, BLEU_4)
                bleu_percent = bleu_score * 100
                bleu_percent_list.append(bleu_percent)
            bleu_score = corpus_bleu(refs, gens_decoded, weights = [0.25, 0.25, 0.25, 0.25])    # weights : 일반적으로 0.25씩 가중치를 두는 듯? 적어도 Delete, 
            bleu_percent = bleu_score * 100

            # GLEU    
            gleu_score = corpus_gleu(refs, gens_decoded, min_len = 1, max_len = 4)
            gleu_percent = gleu_score * 100

            print('model_name : {},\n BLEU_score : {} / BLEU_by_n-gram : {},\n GLEU_score : {}'.format(args.model, bleu_percent, bleu_percent_list, gleu_percent))

        elif args.metric == 'PPL':

            from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
            '''
            If you first load GPT2_LM from huggingface reopo, save the weights at local directory.
            '''
            # hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # GPT2_LM = TFGPT2LMHeadModel.from_pretrained('gpt2')
            # MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM'
            # if os.path.exists(MODEL_SAVE_PATH):
            #     print(f"{MODEL_SAVE_PATH} -- Folder already exists \n")
            # else:
            #     os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            #     print(f"{MODEL_SAVE_PATH} -- Folder create complete \n")
            # # save tokenizer, model
            # GPT2_LM.save_pretrained(MODEL_SAVE_PATH)
            # hf_gpt2_tokenizer.save_pretrained(MODEL_SAVE_PATH)
            '''
            if you already have weights for GPT2_LM at your local directory, then just load it.
            '''
            # Load Fine-tuning model
            gpt_lr = input('gpt_lr=')
            gpt2_lm_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM_finetune' + str('_lr=' + gpt_lr)
            # gpt2_lm_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM_finetune_epoch=10_batch_size=128' + str('_lr=' + gpt_lr)
            # hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_lm_dir)
            hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            hf_gpt2_tokenizer.pad_token = '[pad]'
            GPT2_LM = TFGPT2LMHeadModel.from_pretrained(gpt2_lm_dir)

            # 허깅페이스 사전 (= hf_token_dict) 구축
            hf_token_dict = {v:k for k,v in hf_gpt2_tokenizer.get_vocab().items()}

            # 원형태의 생성 시퀀스 (= gens_decoded)를 허깅페이스 사전을 통해 토크나이징; padding = True를 통해 시퀀스 길이 맞춰주기
            gens_txt = list(map(lambda x : " ".join(x), gens_decoded))
            encoded_txt = hf_gpt2_tokenizer(gens_txt, return_tensors='tf', padding=True)
            print('gens_txt :', gens_txt[:50])
            print('gens_decoded_string :', gens_decoded_string[:50])


            # 토크나이징된 시퀀스를 인풋 시퀀스와 아웃풋 시퀀스로 구분
            encoded_inputs = encoded_txt['input_ids'][:, :-1]
            encoded_outputs = encoded_txt['input_ids'][:, 1:]

            # GPT2를 통해 시퀀스 내 각 토큰들의 다음 토큰에 대한 로짓 예측
            pred_outputs = GPT2_LM(encoded_inputs)
            gc.collect()

            # 시퀀스 최대길이 추출
            maxlen = tf.shape(encoded_txt['input_ids'])[1]

            # with tf.device("/cpu:0"):
            #     # 소프트맥스를 통해 다음 토큰에 대한 로짓값을 확률분포로 변환
            #     pred_probs = tf.nn.softmax(pred_outputs.logits, axis = -1)

            #     # 아웃풋 시퀀스를 원핫 인코딩 변환 (아래에서 pred_probs와 multiply() 해주기 위함.)
            #     target_onehot = tf.one_hot(encoded_outputs, depth = len(hf_token_dict))

            #     # 다음 토큰의 예측 확률분포 (pred_probs)와 실제 다음 토큰 (target_onehot)를 곱하여 실제 다음 토큰의 확률값만 남긴 후, 해당 토큰 값을 추출
            #     sparse_pre_probs = tf.math.multiply(pred_probs, target_onehot)
            #     pred_tokens = tf.argmax(sparse_pre_probs, axis =-1)
            #     gc.collect()

            #     # [pad] 토큰들을 0으로 채운 마스크 행렬
            #     pad_token_mask = tf.cast(tf.math.not_equal(pred_tokens, hf_gpt2_tokenizer.pad_token_id), dtype = tf.float32)

            #     # 실제 다음 토큰들의 예측 확률 값들을 시간축을 따라 누적곱 한 뒤, 마스크 행렬을 통해 일부 time-step에 대해서는 누적곱의 값을 0으로 강제
            #     cumul_token_prods = tf.math.cumprod(tf.reduce_sum(sparse_pre_probs, axis = -1), axis = -1)
            #     padded_cumul_token_prods = cumul_token_prods * pad_token_mask
            #     # print('padded_cumul_token_probs :', padded_cumul_token_prods)

            #     # 각 시퀀스별로 최초로 0이 등장하는 time-step (= 최초로 마스크 토큰이 등장하거나 누적곱 값이 너무 작아서져 최초로 0으로 수렴한 time-step) 바로 이전 time-step에 대한 인덱스 추출
            #     cumul_prob_idx = tf.cast(tf.argmin(padded_cumul_token_prods, axis = -1)-1, dtype = tf.int32)
            #     target_idx = tf.concat([tf.range(len(cumul_prob_idx))[:, tf.newaxis], cumul_prob_idx[:, tf.newaxis]], axis = 1)
            #     gc.collect()
            #     # print('cumul_prob_idx :', cumul_prob_idx)

            #     # 추출한 인덱스 (target_idx)에 속한 값, 즉 각 시퀀스별로 토큰의 예측확률의 누적곱이 0이 아닌 마지막 값을 추출(= final_cumul_prob)
            #     final_cumul_prob = tf.gather_nd(params=padded_cumul_token_prods, indices=target_idx)
            #     # print('final_cumul_prob :', final_cumul_prob)

            #     # [pad] 토큰이 아닌 토큰들의 갯수 (= 시퀀스의 길이)를 계산하기 위해, 
            #     # pad_token_mask에서 첫번째로 0값이 등장한 (= 첫번째로 [pad]토큰이 등장한) 컬럼 인덱스를 reduce_sum()을 통해 계산
            #     first_pad = tf.reduce_sum(pad_token_mask, axis = -1)
            #     # print('first_pad : ', first_pad)

            #     # 앞서 구한 final_cumul_prob를 가지고 공식을 따라 PPL 스코어 계산
            #     ppl_score = final_cumul_prob ** tf.cast(-1/(maxlen), dtype = tf.float32)
            #     # ppl_score = final_cumul_prob ** tf.cast(-1/(cumul_prob_idx + 1), dtype = tf.float32)
            #     # ppl_score = final_cumul_prob ** tf.cast(-1/(first_pad), dtype = tf.float32)
            #     mean_ppl_score = tf.reduce_mean(ppl_score)
            # print('model_name : {},\n PPL_score : {}, max_seqlen : {}'.format(args.model, mean_ppl_score, np.max(first_pad)))

            SCCE_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mean_ppl_score = tf.reduce_mean(SCCE_loss(encoded_outputs, pred_outputs.logits))
            print('model_name : {},\n PPL_score : {}'.format(args.model, mean_ppl_score))


    elif args.model == 'baseline':
        '''
        baseline 평가
        '''
        os.chdir('/home/messy92/Leo/NAS_folder/ICML23/proposed')
        # baseline_name = 'DIRR-CA'
        baseline_name = input('baseline_name : ')   # CA, DL, LatentEdit, UNMT, CycleRL, DualRL, DIRR-CA, DIRR-CO, 
                                                    # DO, RO, DR, TB, B-GST, G-GST, mult_dis, cond_dis, T&G, LEWIS
        gens_decoded = []
        for idx, transfer_direction in enumerate(['neg_to_pos', 'pos_to_neg']):
            target_file_name = transfer_direction + ' (' + baseline_name + ').txt'
            target_file_dir = find_all(target_file_name, os.getcwd())
            baseline_gens_txt = [line.split() for line in open(target_file_dir[0])] 

            if idx == 0:
                gens_decoded = baseline_gens_txt
            else:
                gens_decoded += baseline_gens_txt

        if args.metric == 'ACC':
            m = tf.keras.metrics.CategoricalAccuracy()
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
            '''
            If you first load distil_BERT from huggingface ropo, save the weights at local directory.
            '''
            # checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
            # hf_bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            # distil_BERT = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
            # MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/Distil_BERT'
            # if os.path.exists(MODEL_SAVE_PATH):
            #     print(f"{MODEL_SAVE_PATH} -- Folder already exists \n")
            # else:
            #     os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            #     print(f"{MODEL_SAVE_PATH} -- Folder create complete \n")
            # # save tokenizer, model
            # distil_BERT.save_pretrained(MODEL_SAVE_PATH)
            # hf_bert_tokenizer.save_pretrained(MODEL_SAVE_PATH)
            '''
            if you already have weights for distil_BERT at your local directory, then just load it.
            '''
            # Load Fine-tuning model
            # bert_lr = input('bert_lr=')
            # distil_bert_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/Distil_BERT_epoch=10_batch_size=512' + str('_lr=' + bert_lr)
            # hf_bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            distil_bert_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/yelp_Base_BERT'
            hf_bert_tokenizer = AutoTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
            hf_bert_tokenizer.pad_token = '[pad]'
            distil_BERT = TFAutoModelForSequenceClassification.from_pretrained(distil_bert_dir)

            # 허깅페이스 distilbert 기반 분류기 활용
            gens_decoded_string = [" ".join(a_gen) for a_gen in gens_decoded]
            gens_decoded_string = hf_bert_tokenizer(gens_decoded_string, padding=True, truncation=True, return_tensors="tf")
            pred_outputs = distil_BERT(gens_decoded_string, training = False)
            # pred_probs = tf.nn.softmax(pred_outputs.logits, axis = -1)
            rev_attr_onehot = tf.one_hot(1 - test_attribute, depth = len(tf.unique(test_attribute)[0]))
            # mean_score = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(pred_probs, rev_attr_onehot), axis = -1))
            m.update_state(rev_attr_onehot, pred_outputs.logits)

            # print('baseline_name : {},\n ACC : {}'.format(baseline_name, mean_score.numpy()))
            print('baseline_name : {},\n ACC : {}'.format(baseline_name, m.result().numpy()))

        elif args.metric == 'BLEU':
            from nltk.translate.bleu_score import corpus_bleu
            from nltk.translate.gleu_score import corpus_gleu

            references = [[a] for a in references]
            bleu_percent_list = []
            for i in range(4):
                n_gram_setup = [0, 0, 0, 0]
                n_gram_setup[i] = 1
                bleu_score = corpus_bleu(references, gens_decoded, weights = n_gram_setup)    # weights : n-gram 조절 (BLEU_1, BLEU_2, BLEU_3, BLEU_4)
                bleu_percent = bleu_score * 100
                bleu_percent_list.append(bleu_percent)
            bleu_score = corpus_bleu(references, gens_decoded, weights = [0.25, 0.25, 0.25, 0.25])    # weights : 일반적으로 0.25씩 가중치를 두는 듯? 적어도 Delete, 
            bleu_percent = bleu_score * 100

            # GLEU    
            gleu_score = corpus_gleu(references, gens_decoded, min_len = 1, max_len = 4)
            gleu_percent = gleu_score * 100

            print('baseline_name : {},\n BLEU_score : {} / BLEU_by_n-gram : {},\n GLEU_score : {}'.format(baseline_name, bleu_percent, bleu_percent_list, gleu_percent))

        elif args.metric == 'PPL':

            from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
            '''
            If you first load GPT2_LM from huggingface ropo, save the weights at local directory.
            '''
            # hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # GPT2_LM = TFGPT2LMHeadModel.from_pretrained('gpt2')
            # MODEL_SAVE_PATH = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM'
            # if os.path.exists(MODEL_SAVE_PATH):
            #     print(f"{MODEL_SAVE_PATH} -- Folder already exists \n")
            # else:
            #     os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            #     print(f"{MODEL_SAVE_PATH} -- Folder create complete \n")
            # # save tokenizer, model
            # GPT2_LM.save_pretrained(MODEL_SAVE_PATH)
            # hf_gpt2_tokenizer.save_pretrained(MODEL_SAVE_PATH)
            '''
            if you already have weights for GPT2_LM at your local directory, then just load it.
            '''
            # Load Fine-tuning model
            gpt_lr = input('gpt_lr=')
            gpt2_lm_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM_finetune' + str('_lr=' + gpt_lr)
            # gpt2_lm_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/GPT2LM_finetune_epoch=10_batch_size=128' + str('_lr=' + gpt_lr)
            # hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_lm_dir)
            hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            hf_gpt2_tokenizer.pad_token = '[pad]'
            GPT2_LM = TFGPT2LMHeadModel.from_pretrained(gpt2_lm_dir)

            # 허깅페이스 사전 (= hf_token_dict) 구축
            hf_token_dict = {v:k for k,v in hf_gpt2_tokenizer.get_vocab().items()}

            # 원형태의 생성 시퀀스 (= gens_decoded)를 허깅페이스 사전을 통해 토크나이징; padding = True를 통해 시퀀스 길이 맞춰주기
            gens_txt = list(map(lambda x : " ".join(x), gens_decoded))
            encoded_txt = hf_gpt2_tokenizer(gens_txt, return_tensors='tf', padding=True)

            # 토크나이징된 시퀀스를 인풋 시퀀스와 아웃풋 시퀀스로 구분
            encoded_inputs = encoded_txt['input_ids'][:, :-1]
            encoded_outputs = encoded_txt['input_ids'][:, 1:]

            # GPT2를 통해 시퀀스 내 각 토큰들의 다음 토큰에 대한 로짓 예측
            pred_outputs = GPT2_LM(encoded_inputs)
            gc.collect()

            # 시퀀스 최대길이 추출
            maxlen = tf.shape(encoded_txt['input_ids'])[1]

            # with tf.device("/cpu:0"):
            #     # 소프트맥스를 통해 다음 토큰에 대한 로짓값을 확률분포로 변환
            #     pred_probs = tf.nn.softmax(pred_outputs.logits, axis = -1)

            #     # 아웃풋 시퀀스를 원핫 인코딩 변환 (아래에서 pred_probs와 multiply() 해주기 위함.)
            #     target_onehot = tf.one_hot(encoded_outputs, depth = len(hf_token_dict))

            #     # 다음 토큰의 예측 확률분포 (pred_probs)와 실제 다음 토큰 (target_onehot)를 곱하여 실제 다음 토큰의 확률값만 남긴 후, 해당 토큰 값을 추출
            #     sparse_pre_probs = tf.math.multiply(pred_probs, target_onehot)
            #     pred_tokens = tf.argmax(sparse_pre_probs, axis =-1)
            #     gc.collect()

            #     # [pad] 토큰들을 0으로 채운 마스크 행렬
            #     pad_token_mask = tf.cast(tf.math.not_equal(pred_tokens, hf_gpt2_tokenizer.pad_token_id), dtype = tf.float32)

            #     # 실제 다음 토큰들의 예측 확률 값들을 시간축을 따라 누적곱 한 뒤, 마스크 행렬을 통해 일부 time-step에 대해서는 누적곱의 값을 0으로 강제
            #     cumul_token_prods = tf.math.cumprod(tf.reduce_sum(sparse_pre_probs, axis = -1), axis = -1)
            #     padded_cumul_token_prods = cumul_token_prods * pad_token_mask
            #     # print('padded_cumul_token_probs :', padded_cumul_token_prods)

            #     # 각 시퀀스별로 최초로 0이 등장하는 time-step (= 최초로 마스크 토큰이 등장하거나 누적곱 값이 너무 작아서져 최초로 0으로 수렴한 time-step) 바로 이전 time-step에 대한 인덱스 추출
            #     cumul_prob_idx = tf.cast(tf.argmin(padded_cumul_token_prods, axis = -1)-1, dtype = tf.int32)
            #     target_idx = tf.concat([tf.range(len(cumul_prob_idx))[:, tf.newaxis], cumul_prob_idx[:, tf.newaxis]], axis = 1)
            #     gc.collect()
            #     # print('cumul_prob_idx :', cumul_prob_idx)

            #     # 추출한 인덱스 (target_idx)에 속한 값, 즉 각 시퀀스별로 토큰의 예측확률의 누적곱이 0이 아닌 마지막 값을 추출(= final_cumul_prob)
            #     final_cumul_prob = tf.gather_nd(params=padded_cumul_token_prods, indices=target_idx)
            #     # print('final_cumul_prob :', final_cumul_prob)

            #     # [pad] 토큰이 아닌 토큰들의 갯수 (= 시퀀스의 길이)를 계산하기 위해, 
            #     # pad_token_mask에서 첫번째로 0값이 등장한 (= 첫번째로 [pad]토큰이 등장한) 컬럼 인덱스를 reduce_sum()을 통해 계산
            #     first_pad = tf.reduce_sum(pad_token_mask, axis = -1)

            #     # 앞서 구한 final_cumul_prob를 가지고 공식을 따라 PPL 스코어 계산
            #     ppl_score = final_cumul_prob ** tf.cast(-1/(maxlen), dtype = tf.float32)
            #     # ppl_score = final_cumul_prob ** tf.cast(-1/(cumul_prob_idx + 1), dtype = tf.float32)
            #     # ppl_score = final_cumul_prob ** tf.cast(-1/(first_pad), dtype = tf.float32)
            #     mean_ppl_score = tf.reduce_mean(ppl_score)
            # print('model_name : {},\n PPL_score : {}, max_seqlen : {}'.format(args.model, mean_ppl_score, np.max(first_pad)))

            SCCE_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mean_ppl_score = tf.reduce_mean(SCCE_loss(encoded_outputs, pred_outputs.logits))
            print('model_name : {},\n PPL_score : {}'.format(args.model, mean_ppl_score))

elif args.task == 'DR':

    '''
    학습 데이터는 22개의 disease와 상호작용하는 13933개의 compound로 구성된 총 27667쌍의 compound-disease 상호작용 데이터.
    구체적으로, 단일한 disease가 복수의 compound와 매칭되거나 단일한 compound가 복수의 disease에 매칭되는 many-to-many 구조를 가짐.
    위와 같은 many-to-many 구조를 토대로 protein sequence로부터 특정

    Original disease에 해당하는 Protein Sequence와 Secondary disaese에 해당하는 Disease Code를 인풋으로 입력하여 Compound를 생성
    생성된 Compound가 Secondary disease에 유효한 실제 Compound와 거의 일치하는 Compound가 생성되는지 여부를 확인하고자 함
    '''
    # 경로 재설정
    os.chdir('/home/messy92/Leo/NAS_folder/ICML23')


    '''
    test_input_sequence 및 test_output_sequence 데이터 구축    
    '''
    # 약물 메타정보 데이터 로드
    drug_dat = pd.read_csv('./data/drug-discovery/gyumin/Repositioned_edited.csv')          # 41 x 7    

    # 화합물 샘플 데이터 로드
    sample_dat = pd.read_csv('./data/drug-discovery/gyumin/BindingDB_repositioned.csv')     # 51834 x 6

    # 약물-x 화합물 데이터 구축
    drug_x_sample_dat = pd.merge(drug_dat, sample_dat, left_on = 'Sequence', right_on = 'Target_seq',how = 'inner')

    # 토크나이저 로드
    with open('./prep_data/drug-discovery/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    # 질병 코드맵 로드
    with open('./prep_data/drug-discovery/disease_code_map.pickle', 'rb') as f:
        disease_code_map = pickle.load(f)

    # original/secondary drug 보기
    drug_dat = pd.read_csv('./data/drug-discovery/gyumin/Repositioned_edited.csv')
    drug_reposition_dat = drug_dat.groupby(['Related drug', 'Related disease', 'Indication']).describe()['Sequence']['top'].reset_index()
    drug_reposition_dat.to_csv('/home/messy92/Leo/NAS_folder/ICML23/drug_reposition.csv')
   
    # 토큰 사전 가져오기
    with open('/home/messy92/Leo/NAS_folder/ICML23/prep_data/drug-discovery' + '/token_dict.pickle', 'rb') as f:
        token_dict = pickle.load(f)
    special_token_list = ['[pad]', '[mask]']
    test_attribute = np.load('./prep_data/drug-discovery/attribute(test).npy')
    reward_class_token_list = ['[' + 'R_' + str(reward_class) + ']' for reward_class in range(len(np.unique(test_attribute)))]
    edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
    add_token_list = special_token_list + reward_class_token_list + edit_token_list
    token_dict = add_token_list_in_dict(add_token_list, token_dict)
    action_set = list(token_dict.values())[-len(edit_token_list):]

    # (reposition이 알려진) compound의 original/secondary 타겟 단백질 시퀀스 및 indication code 구축
    repositioned_disease_pair, repositioned_code_pair, ori_reposition_seq_code_pair, sec_reposition_seq_code_pair = get_reposition_pair(drug_reposition_dat, disease_code_map)
    ori_reposition_seq_code_pair_df = pd.DataFrame(ori_reposition_seq_code_pair, columns = ['Target_Seq', 'Indication_Code'])
    sec_reposition_seq_code_pair_df = pd.DataFrame(sec_reposition_seq_code_pair, columns = ['Target_Seq', 'Indication_Code'])
    ori_target_protein_seq = ori_reposition_seq_code_pair_df['Target_Seq']
    ori_ind_code = ori_reposition_seq_code_pair_df['Indication_Code']
    sec_target_protein_seq = sec_reposition_seq_code_pair_df['Target_Seq']
    sec_ind_code = sec_reposition_seq_code_pair_df['Indication_Code']

    # Successful Reposition Case에 해당하는 original indication의 단백질 시퀀스 및 해당 시퀀스와 상호작용 (interaction)하는 화합물 시퀀스 데이터 추출
    ori_target_interaction_dat, ori_ind_code_list = filter_interaction_dat(protein_seq=ori_target_protein_seq, ind_code=ori_ind_code, interaction_dat=drug_x_sample_dat)

    # indication code 컬럼을 생성
    ori_target_interaction_dat['Ori_ind_code'] = ori_ind_code_list

    # Secondary Indication에 해당하는 Code 컬럼 생성하기
    sec_ind_array = get_second_ind_code_vector(
                                                ori_interaction_dat = ori_target_interaction_dat, 
                                                ori_ind_code_list = ori_ind_code_list, 
                                                code_reposition_map = repositioned_code_pair
                                                )
    ori_target_interaction_dat['Sec_ind_code'] = sec_ind_array

    # 토크나이징 및 {test_input, test_output}_sequence 선언
    test_input_sequence = ori_target_interaction_dat['Sequence']
    test_input_sequence_encoded = tokenizer.texts_to_sequences(test_input_sequence.apply(lambda x: ['[BOS]'] + list(x) + ['[EOS]']))
    test_input_sequence = pad_sequences(test_input_sequence_encoded, padding='post')
    test_output_sequence = ori_target_interaction_dat['Compound_smiles']
    test_output_sequence_encoded = tokenizer.texts_to_sequences(test_output_sequence.apply(lambda x: ['[BOS]'] + list(x) + ['[EOS]']))
    test_output_sequence = pad_sequences(test_output_sequence_encoded, padding='post')

    '''
    프롬프트 벡터 (= 보상 클래스 토큰) 생성 준비    
    '''
    # Seoconary Indication Disease에 해당하는 token 생성
    ori_target_interaction_dat_copy = copy.deepcopy(ori_target_interaction_dat)
    ori_target_interaction_dat_copy['Sec_ind_token'] = ori_target_interaction_dat_copy['Sec_ind_code']
    for code in sec_ind_code:

        sec_ind_disease_label = '[R_' + str(code) + ']'
        print('sec_ind_disease_label :', sec_ind_disease_label)
        sec_ind_disease_token = get_token(token_dict, sec_ind_disease_label)

        target_idx = ori_target_interaction_dat_copy[ori_target_interaction_dat_copy['Sec_ind_code'] == code].index
        ori_target_interaction_dat_copy.loc[target_idx, 'Sec_ind_token'] = sec_ind_disease_token

    # 프롬프트 벡터 생성
    test_reward_class_vector = np.array(ori_target_interaction_dat_copy['Sec_ind_token'])
    test_prompt_vector = test_reward_class_vector[:, np.newaxis]

    '''
    test_target_sequence 데이터 구축    
    --> test_target_sequence는 test_input_sequence의 Secondary_indication에 해당하는 disease와 관련된 compound seuqence로 정의
    '''
    # test_target_sequence 구축
    sec_target_interaction_dat, _ = filter_interaction_dat(protein_seq=sec_target_protein_seq, ind_code=sec_ind_code, interaction_dat=drug_x_sample_dat)
    test_target_sequence = sec_target_interaction_dat['Compound_smiles']
    test_target_sequence_encoded = tokenizer.texts_to_sequences(test_target_sequence.apply(lambda x: ['[BOS]'] + list(x) + ['[EOS]']))
    test_target_sequence = pad_sequences(test_target_sequence_encoded, padding='post')
    
    # BLEU 계산용 refereces 데이터셋 만들기
    references = get_decoded_list(test_target_sequence, token_dict)
    references = list(map(''.join, references))

    '''
    attribute 데이터 로드    
    '''
    # control_attribute는 test_prompt_vector에 의해 얼마나 약물생성이 잘 제어되었는지 평가하기 위해 필요.
    # reward_model에 gen_sequence를 입력하면 control_attribute에 해당하는 label의 예측값이 가장 높아야 함
    control_attribute = copy.deepcopy(sec_ind_array)


    if args.model == 'proposed':
        # target_task = args.task                 # {ST, DR}
        # decoding_mode = input('decoding= ')     # {AR, NAR}
        # num_epochs = input('num_epochs=')
        # batch_size = input('batch_size=')
        # lr = input('lr=')
        # opt_type = input('opt=')                # {None, CosDecay, CosDecayRe, ExpDecay, InvTimeDecay}
        # len_buffer = input('len_buffer(lb)= ')
        # eta = input('eta=')
        # es = input('es=')                       # {greedy, stochastic}
        # reward_type = input('reward=')          # {S, A, G, H}
        # algo_type = input('algo=')              # {None, PG, PPO}
        # early_stop = input('early_stop=')

        target_task = 'DR'
        decoding_mode = 'NAR'
        num_epochs = str(30)
        batch_size = str(64)
        lr = str(5e-4)
        opt_type = str(None)
        len_buffer = str(0)
        eta = str(0.005)
        es = 'greedy'
        reward_type = 'G'
        algo_type = 'PG'
        early_stop = 'no'

        # Dataset 객체 자체는 cpu에 할당
        with tf.device("/cpu:0"):
            dataset = tf.data.Dataset.from_tensor_slices((test_input_sequence, test_output_sequence, test_prompt_vector, control_attribute))
            batchset = dataset.batch(batch_size = 32, num_parallel_calls = 8)
            batchset = batchset.prefetch(1)

        # ---------------------------------------------------------------------------------------- #
        # 2-1) 에이전트 모델 초기화
        leva_kwargs_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/drug-discovery/LEVA/kwargs_LEVA' + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
        with open(leva_kwargs_dir, 'r') as f:
            LEVA_kwargs = json.load(f)
        lev_agent = LEVAgent(**LEVA_kwargs)

        # --> 초기 정책모델 학습 가중치 로드하여 타겟 정책모델 가중치 초기화
        if algo_type != "None":
            load_dir2 = '/home/messy92/Leo/NAS_folder/ICML23/weights/drug-discovery/LEVA_' + str(target_task + '_epoch=' + num_epochs + '_opt=' + opt_type + '_lr=' + lr + '_lb=' + len_buffer + '_eta=' + eta + '_es=' + es + '_reward=' + reward_type + '_algo=' + algo_type + '_early_stop=' + early_stop)
        else:
            load_dir2 = '/home/messy92/Leo/NAS_folder/ICML23/weights/drug-discovery/LEVA_' + str(target_task + '_epoch=' + num_epochs + '_opt=' + opt_type + '_lr=' + lr + '_lb=' + len_buffer + '_eta=' + eta + '_es=' + es + '_reward=' + reward_type)

        print('lev_agent_weights_dir : ' , load_dir2)
        lev_agent.load_weights(tf.train.latest_checkpoint(load_dir2))

        # 2-2) 환경 모델 초기화
        env_gen_model_name = 'NAR'
        # env_gen_model_name = 'coBART'

        prefix = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/drug-discovery/' + env_gen_model_name
        env_gen_kwargs = '/kwargs_' + env_gen_model_name + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
        env_gen_kwargs_dir = prefix + env_gen_kwargs
        with open(env_gen_kwargs_dir, 'r') as f:
            env_kwargs = json.load(f)

        env_gen_model = AETransformer(**env_kwargs)

        # --> 초기 환경 모델 학습 가중치 로드하여 타겟 정책모델 가중치 초기화
        prefix = '/home/messy92/Leo/NAS_folder/ICML23/weights/drug-discovery/'
        env_gen_weights_dir = prefix + env_gen_model_name + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
        print(env_gen_weights_dir)
        env_gen_model.load_weights(tf.train.latest_checkpoint(env_gen_weights_dir))

        # ---------------------------------------------------------------------------------------- #
        # 5) 훈련 루프

        # --> 메트릭 초기화
        metrics_names = ['lev_loss', 'div_score', 'reward']
        pre_reward = 0
        reward_history = []
        gens_decoded_all = []
        outputs_decoded_all = []

        # --> 루프 수행
        for idx, (inputs, outputs, attr_tokens, attrs) in enumerate(batchset):        

            enc_pad_mask_for_agent, _, _ = env_gen_model.mask_generator(outputs, outputs)

            '''
            편집 계획 (edit_plans) 생성
            '''
            agent_outputs, action_att_weights = lev_agent(outputs, enc_pad_mask_for_agent, training = False)
            agent_actions, action_tokens = return_action_token(agent_outputs, outputs, token_dict, mode = 'test')       # agent_actions = 에이전트가 선택한 action : (batch_size, seq_len, action_size)
                                                                                                                    # action_tokens = action의 토큰 : (batch_size, seq_len)
            
            '''
            레반슈타인 연산 수행
            '''
            # 레반슈타인 연산자 적용
            masked_outputs, new_action_tokens = apply_lev_operation(outputs, action_tokens, token_dict)
            masked_outputs = masked_outputs.numpy()                       # numpy indexing을 위해 자료형을 텐서에서 넘파이로 바꿔주기
            masked_outputs_copy = copy.deepcopy(masked_outputs)           # masked_outputs는 아래에서 inference 과정동안 계속 update 될 예정이므로 masked_inputs_copy를 따로 만들어 놓기.

            # AutoRegressive 디코딩
            if decoding_mode == 'AR':

                '''
                스타일 전환 : attribute 코드 prefix하기
                '''
                bos_vector = masked_outputs[:, :1]    # [bos] 토큰 벡터
                gen_seqs = np.concatenate([attr_tokens, bos_vector], axis = -1)

                for t in range(masked_outputs.shape[1]-2):

                    # t가 0일 때
                    if t == 0:
                        # 인코더 PAD 마스킹
                        enc_pad_mask_for_env, _, _ = env_gen_model.mask_generator(inputs, inputs)

                        # 인코더 신경망
                        enc_outputs, _, _ = env_gen_model.encoder(inputs, enc_pad_mask_for_env, training = False)        # enc_outputs : (batch_size, seq_len, d_model)

                    # t가 0보다 클 때
                    else:
                        # inputs 업데이트
                        inputs[:, :gen_seqs.shape[1]-1] = gen_seqs[:, 1:]    # gen_seqs에서 첫번째 토큰인 prompt 토큰은 제외해야 하므로 gen_seqs[:, 1:]로 셋팅= ['bos'] 부터 시작.
                                                                                    # gen_seqs에서 첫번째 토큰을 제외했으므로, inputs에 삽입되는 gen_seqs의 shape를 정의해줄 떄 shape[1]-1 해준다.

                        # 인코더 PAD 마스킹
                        enc_pad_mask_for_env, _, _ = env_gen_model.mask_generator(inputs, inputs)

                        # 인코더 신경망
                        enc_outputs, _, _ = env_gen_model.encoder(inputs, enc_pad_mask_for_env, training = False)        # enc_outputs : (batch_size, seq_len, d_model)


                    # 디코더 Subsequent 마스킹 (= Future 마스킹)
                    _, dec_pad_mask, dec_subseq_mask = env_gen_model.mask_generator(masked_outputs, gen_seqs)

                    # 디코더 신경망
                    dec_outputs, _ = env_gen_model.decoder(gen_seqs, enc_outputs, dec_pad_mask, dec_subseq_mask, training = False)       # dec_outputs : (batch_size, seq_len, voca_size)

                    # 토큰 예측
                    pred_tokens = tf.cast(tf.math.argmax(dec_outputs[:, -1, :], axis = -1), dtype = tf.int32)[:, tf.newaxis]

                    # 시퀀스 생성
                    gen_seqs = tf.concat([gen_seqs, pred_tokens], axis = -1)
                    gen_seqs = gen_seqs.numpy()


            # Non-AutoRegressive 디코딩
            elif decoding_mode == 'NAR':

                '''
                스타일 전환 : attribute 코드를 prefix하여 환경 모델에 입력
                '''
                # 마스크 인풋 시퀀스의 앞에 프롬프트 붙여주기
                masked_prompt_outputs = np.concatenate([attr_tokens, masked_outputs], axis = -1)

                # 토큰 예측
                _ ,_, dec_outputs, _ = env_gen_model((inputs, masked_prompt_outputs), training = False)

                # 시퀀스 생성
                gen_seqs = tf.cast(tf.math.argmax(dec_outputs, axis = -1), dtype = tf.int32)
                gen_seqs = gen_seqs.numpy()


            '''
            제어된 시퀀스 생성
            '''
            controlled_gens = fill_pad_after_eos(gen_seqs, masked_outputs_copy, token_dict)
            if idx == 0:
                controlled_gens_all = controlled_gens.numpy()
                attrs_all = attrs
                outputs_all = outputs.numpy()
            else:
                controlled_gens_all = tf.concat([controlled_gens_all, controlled_gens], axis = 0)
                attrs_all = tf.concat([attrs_all, attrs], axis = -1)
                outputs_all = tf.concat([outputs_all, outputs], axis = 0)

        gens_decoded = get_decoded_list(controlled_gens_all[:, 1:], token_dict)
        gens_decoded_all += list(map(''.join, gens_decoded))
        outputs_decoded = get_decoded_list(outputs_all, token_dict)
        outputs_decoded_all += list(map(''.join, outputs_decoded))

        for my_idx in range(0, 5):
            print('attrs : {}'.format(attrs[my_idx]))
            print('inputs : {}'.format([token_dict[t] for t in inputs[my_idx, :].numpy()]))
            print('edit plans : {}'.format([token_dict[t] for t in action_tokens[my_idx, :].numpy()]))
            print('gens : {}'.format([token_dict[t] for t in controlled_gens[my_idx, :].numpy()]))
            print('\n')


        del env_gen_model, lev_agent    

        if args.metric == 'ACC':

            m = tf.keras.metrics.SparseCategoricalAccuracy()

            # 2-1) 에이전트 모델 초기화
            rf_kwargs_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/drug-discovery/Reward_Function/kwargs_attr_reward_function_DR_100_64'
            with open(rf_kwargs_dir, 'r') as f:
                rf_kwargs = json.load(f)
            reward_function = Reward_Function(len(np.unique(test_attribute)), **rf_kwargs)
            reward_function.load_weights(tf.train.latest_checkpoint('/home/messy92/Leo/NAS_folder/ICML23/weights/drug-discovery/attr_reward_function_DR_100_64'))
            enc_pad_mask, _, _ = reward_function.mask_generator(controlled_gens_all[:, 1:], controlled_gens_all[:, 1:])
            attr_logits = reward_function(controlled_gens_all[:, 1:], enc_pad_mask, training = False)
            
            m.update_state(attrs_all, attr_logits)
            m.result()
            
            # print('model_name : {},\n ACC : {}'.format(args.model, mean_score.numpy()))
            print('model_name : {},\n ACC : {}'.format(args.model, m.result().numpy()))

            pred_label = tf.cast(tf.argmax(attr_logits, axis = -1), dtype = tf.int32)
            hit_attr = attrs_all - pred_label
            hit_idx = np.where(hit_attr == 0)[0]
            gens_decoded_hit = np.array(gens_decoded_all)[hit_idx]
            outputs_decoded_hit = np.array(outputs_decoded_all)[hit_idx]
            attrs_all.numpy()[hit_idx]
            base_edit_df = pd.DataFrame({'base' : outputs_decoded_hit, 'edit' : gens_decoded_hit, 'target_ind_code' : attrs_all})
            base_edit_df.to_csv('/home/messy92/Leo/NAS_folder/ICML23/base_edit.csv')

        elif args.metric == 'BLEU':
            from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
            from nltk.translate.gleu_score import corpus_gleu

            refs = [[a] for a in outputs_decoded_all]
            bleu_score = corpus_bleu(refs, gens_decoded_all, weights = [1, 0, 0, 0]) 

            print('model_name : {},\n BLEU_score : {}'.format(args.model, bleu_score))


# save_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/drug-discovery/Distil_BERT'
# distil_BERT.save_weights(save_dir + '/weights.ckpt')

# '''
# # PPL 점수 구하기 (GPT2 기반)
# '''
# # 허깅페이스 사전 (= hf_token_dict) 구축
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
# hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# hf_gpt2_tokenizer.pad_token = '[pad]'
# hf_token_dict = {v:k for k,v in hf_gpt2_tokenizer.get_vocab().items()}

# # GPT2 정의; 참고로, 이렇게 huggingface로부터 받아오는 from_pretrained 모델들은 training = False 모드가 디폴트로 설정되어 있음.
# GPT2_LM = TFGPT2LMHeadModel.from_pretrained('gpt2')

# # 원형태의 생성 시퀀스 (= candidates)를 허깅페이스 사전을 통해 토크나이징; padding = True를 통해 시퀀스 길이 맞춰주기
# gens_txt = list(map(lambda x : " ".join(x), gens_decoded))
# encoded_txt = hf_gpt2_tokenizer(gens_txt, return_tensors='tf', padding = True)

# # 토크나이징된 시퀀스를 인풋 시퀀스와 아웃풋 시퀀스로 구분
# encoded_inputs = encoded_txt['input_ids'][:, :-1]
# encoded_outputs = encoded_txt['input_ids'][:, 1:]

# # GPT2를 통해 시퀀스 내 각 토큰들의 다음 토큰에 대한 로짓 예측
# pred_outputs = GPT2_LM(encoded_inputs)

# # 시퀀스 최대길이 추출
# maxlen = tf.shape(encoded_txt['input_ids'])[1]

# # 소프트맥스를 통해 다음 토큰에 대한 로짓값을 확률분포로 변환
# with tf.device("/cpu:0"):   # If tf.nn.softmax causes OOM memory Error, then execute softmax operation on CPU.
#     pred_probs = tf.nn.softmax(pred_outputs.logits, axis = -1)

# # 아웃풋 시퀀스를 원핫 인코딩 변환 (아래에서 pred_probs와 multiply() 해주기 위함.)
# with tf.device("/cpu:1"):   # If tf.one_hot causes OOM memory Error, then execute softmax operation on CPU.
#     target_onehot = tf.one_hot(encoded_outputs, depth = len(hf_token_dict))
# gc.collect()

# # 다음 토큰의 예측 확률분포 (pred_probs)와 실제 다음 토큰 (target_onehot)를 곱하여 실제 다음 토큰의 확률값만 남긴 후, 해당 토큰 값을 추출
# sparse_pre_probs = tf.math.multiply(pred_probs, target_onehot)
# pred_tokens = tf.argmax(sparse_pre_probs, axis =-1)

# # [pad] 토큰들을 0으로 채운 마스크 행렬
# pad_token_mask = tf.cast(tf.math.not_equal(pred_tokens, hf_gpt2_tokenizer.pad_token_id), dtype = tf.float32)

# # 실제 다음 토큰들의 예측 확률 값들을 시간축을 따라 누적곱 한 뒤, 마스크 행렬을 통해 일부 time-step에 대해서는 누적곱의 값을 0으로 강제
# cumul_token_prods = tf.math.cumprod(tf.reduce_sum(sparse_pre_probs, axis = -1), axis = -1)
# padded_cumul_token_prods = cumul_token_prods * pad_token_mask

# # 각 시퀀스별로 최초로 0이 등장하는 time-step (= 최초로 마스크 토큰이 등장하거나 누적곱 값이 너무 작아서져 최초로 0으로 수렴한 time-step) 바로 이전 time-step에 대한 인덱스 추출
# cumul_prob_idx = tf.cast(tf.argmin(padded_cumul_token_prods, axis = -1)-1, dtype = tf.int32)
# target_idx = tf.concat([tf.range(len(cumul_prob_idx))[:, tf.newaxis], cumul_prob_idx[:, tf.newaxis]], axis = 1)

# # 추출한 인덱스 (target_idx)에 속한 값, 즉 각 시퀀스별로 토큰의 예측확률의 누적곱이 0이 아닌 마지막 값을 추출(= final_cumul_prob)
# final_cumul_prob = tf.gather_nd(params=padded_cumul_token_prods, indices=target_idx)

# # 앞서 구한 final_cumul_prob를 가지고 공식을 따라 PPL 스코어 계산
# ppl_score = final_cumul_prob ** tf.cast(-1/(maxlen-1), dtype = tf.float32)
# mean_ppl_score = tf.reduce_mean(ppl_score)





# # '''
# # # BLEU & GLEU 점수 구하기
# # '''
# # gens = controlled_gens[:, 1:]
# # candidates = get_decoded_list(gens, token_dict)
# # references = get_decoded_list(inputs, token_dict)
# # references = [[a] for a in references]

# # # BLEU
# # bleu_percent_list = []
# # for i in range(4):
# #     n_gram_setup = [0, 0, 0, 0]
# #     n_gram_setup[i] = 1
# #     bleu_score = corpus_bleu(references, candidates, weights = n_gram_setup)    # weights : n-gram 조절 (BLEU_1, BLEU_2, BLEU_3, BLEU_4)
# #     bleu_percent = bleu_score * 100
# #     bleu_percent_list.append(bleu_percent)
# # bleu_score = corpus_bleu(references, candidates, weights = [0.25, 0.25, 0.25, 0.25])    # weights : 일반적으로 0.25씩 가중치를 두는 듯? 적어도 Delete, 


# # # GLEU    
# # gleu_score = corpus_gleu(references, candidates, min_len = 1, max_len = 4)
# # gleu_percent = gleu_score * 100

# # print('bleu_percent_list : {}, \n gleu_percent_list : {}'.format(bleu_percent_list, gleu_percent))

# # # ################################################################################################################################################
# # '''
# # # ACC 구하기 (RoBERTa 기반)
# # '''

# # # 허깅페이스 distilbert 기반 분류기 활용
# # generated_inputs = [" ".join(a_gen) for a_gen in candidates]
# # # generated_inputs = [" ".join(a_gen) for a_gen in baseline_gens_txt_all]

# # from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
# # checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # generated_inputs = tokenizer(generated_inputs, padding=True, truncation=True, return_tensors="tf")
# # checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# # model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
# # outputs = model(generated_inputs, training = False)
# # preds_class = tf.argmax(outputs.logits, axis = -1)
# # neg_to_pos_score = tf.reduce_sum(preds_class[:500])/500
# # pos_to_neg_score = tf.reduce_sum(preds_class[500:])/500
# # mean_score = (neg_to_pos_score + pos_to_neg_score)/2
# # mean_score.numpy()


# # ################################################################################################################################################
# '''
# # PPL 점수 구하기 (GPT2 기반)
# '''
# # 허깅페이스 사전 (= hf_token_dict) 구축
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# hf_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# hf_gpt2_tokenizer.pad_token = '[pad]'
# # hf_gpt2_tokenizer.pad_token = hf_gpt2_tokenizer.eos_token
# # hf_gpt2_tokenizer.add_special_tokens({'pad_token': '[pad]'})
# # hf_gpt2_tokenizer.add_special_tokens({'bos_token': '[bos]', 'eos_token': '[eos]', 'pad_token': '[pad]'})

# # num_default_tokens = len(hf_gpt2_tokenizer.decoder)
# # num_added_tokens = len(hf_gpt2_tokenizer.added_tokens_decoder)
# # hf_token_dict = hf_gpt2_tokenizer.decoder | hf_gpt2_tokenizer.added_tokens_decoder

# hf_token_dict = {v:k for k,v in hf_gpt2_tokenizer.get_vocab().items()}

# # GPT2 모델에 생성시퀀스를 입력한 뒤, 디코더에서 반환되는 예측 확률값들을 누적곱하기
# GPT2_LM = TFGPT2LMHeadModel.from_pretrained('gpt2')             # GPT2 정의 (참고로, training = False 모드가 디폴트임) 
# # GPT2_LM.resize_token_embeddings(len(hf_gpt2_tokenizer))         # 추가된 토큰을 인식할 수 있도록 모델 토큰 임베딩을 리사이즈

# # 원형태의 생성 시퀀스 (= candidates)를 허깅페이스 사전을 통해 토크나이징
# # gens_decoded_with_special_tokens = list(map(lambda x : ['[bos]'] + x + ['[eos]'], gens_decoded))
# gens_txt = list(map(lambda x : " ".join(x), gens_decoded))
# encoded_txt = hf_gpt2_tokenizer(gens_txt, return_tensors='tf', padding = True)

# # 토크나이징된 시퀀스를 인풋 시퀀스와 아웃풋 시퀀스로 구분
# encoded_inputs = encoded_txt['input_ids'][:, :-1]
# encoded_outputs = encoded_txt['input_ids'][:, 1:]

# # GPT2를 통해 시퀀스 내 각 토큰들의 확률 예측
# # text = "When it comes to making transformers easy, HuggingFace is the"
# # encoded_inputs = hf_gpt2_tokenizer(text, return_tensors='tf')
# pred_outputs = GPT2_LM(encoded_inputs)

# # # GPT2_LM 잘 동작하는지 .generate() 메소드로 체크하기
# maxlen = tf.shape(encoded_txt['input_ids'])[1]
# # pred_outputs = GPT2_LM.generate(encoded_txt['input_ids'][:2, :], max_length=maxlen+1)
# # [hf_token_dict[t] for t in pred_outputs[:, ].numpy()[0]]

# with tf.device("/cpu:0"):   # If tf.nn.softmax causes OOM memory Error, then execute softmax operation on CPU.
#     pred_probs = tf.nn.softmax(pred_outputs.logits, axis = -1)
# with tf.device("/cpu:1"):
#     target_onehot = tf.one_hot(encoded_outputs, depth = len(hf_token_dict))
# gc.collect()

# sparse_pre_probs = tf.math.multiply(pred_probs, target_onehot)
# pred_tokens = tf.argmax(sparse_pre_probs, axis =-1)

# # [pad] 토큰들을 0으로 채운 마스크 행렬
# pad_token_mask = tf.cast(tf.math.not_equal(pred_tokens, hf_gpt2_tokenizer.pad_token_id), dtype = tf.float32)
# cumul_token_prods = tf.math.cumprod(tf.reduce_sum(sparse_pre_probs, axis = -1), axis = -1)
# padded_cumul_token_prods = cumul_token_prods * pad_token_mask
# cumul_prob_idx = tf.cast(tf.argmin(padded_cumul_token_prods, axis = -1)-1, dtype = tf.int32)
# target_idx = tf.concat([tf.range(len(cumul_prob_idx))[:, tf.newaxis], cumul_prob_idx[:, tf.newaxis]], axis = 1)
# final_cumul_prob_idx = tf.gather_nd(params=padded_cumul_token_prods, indices=target_idx)
# ppl_score = final_cumul_prob_idx ** tf.cast(-1/(maxlen-1), dtype = tf.float32)
# mean_ppl_score = tf.reduce_mean(ppl_score)




# #################################################################################################################################### 
# new_candidates = []
# for a_candidate in gens_decoded:
#     a_candidate = [hf_gpt2_tokenizer.bos_token] + a_candidate + [hf_gpt2_tokenizer.eos_token]
#     # a_candidate_encoded = list(hf_gpt2_tokenizer.encode(a_candidate, return_tensors='tf').numpy()[0])
#     a_candidate_encoded = hf_gpt2_tokenizer(a_candidate)['input_ids']
#     a_candidate_encoded_flatten = np.concatenate(a_candidate_encoded).tolist()

#     new_candidates.append(a_candidate_encoded_flatten)

# # 허깅페이스 사전으로 토크나이징 된 생성시퀀스 (= new_candidates)에 패딩 추가.
# hf_test_input_sequence = pad_sequences(new_candidates, padding='post', value = get_token(hf_token_dict, '[pad]'))

# # GPT2 모델에 생성시퀀스를 입력한 뒤, 디코더에서 반환되는 예측 확률값들을 누적곱하기
# GPT2_LM = TFGPT2LMHeadModel.from_pretrained('gpt2')             # GPT2 정의 (참고로, training = False 모드가 디폴트임) 

# # PPL 스코어 계산
# # pred_outputs = GPT2_LM.generate(hf_test_input_sequence, max_length=50)
# pred_outputs = GPT2_LM(hf_test_input_sequence['input_ids'])
# pred_probs = pred_outputs.logits


# text = "When it comes to making transformers easy, HuggingFace is the"
# # text = " ".join(gens_decoded[0])
# # encoded_inputs = hf_gpt2_tokenizer(text, return_tensors='tf')
# # pred_outputs = GPT2_LM(encoded_inputs)
# pred_outputs = GPT2_LM(encoded_txt)
# pred_probs = tf.math.softmax(pred_outputs.logits[0, -1, :], axis =-1)
# pred_tokens = tf.math.argmax(pred_probs, axis = -1)
# print(hf_gpt2_tokenizer.decode(pred_tokens))

# # maxlen = tf.shape(encoded_inputs['input_ids'])[1]
# # pred_outputs = GPT2_LM.generate(encoded_inputs, max_length=maxlen+1)
# # [hf_token_dict[t] for t in pred_outputs[:, ].numpy()[0]]

# # # 허깅페이스 사전에 맞추어 직접 디코딩하기
# # # get_decoded_list(hf_test_input_sequence, hf_token_dict)
# # for t in a_candidate_encoded:
# #     if len(t) <= 1:
# #         print(hf_token_dict[t[0]])
# #     else:    
# #         v = []
# #         for p in t:
# #             v += hf_token_dict[p]
# #         print(''.join(v))

# # ################################################################################################################################################

# '''
# 내 모델 생성 결과 txt 저장
# '''
# ddd


# # %%
# # # 원형태의 생성 시퀀스를 허깅페이스 사전을 통해 토크나이징
# # [[get_token(hf_token_dict, t) for t in a_candidate] for a_candidate in candidates]

# # # 가장 긴 candidates의 길이에 맞춰 padding을 적용한 텐서 어레이 생성
# # gens2 = gens.numpy()[:, 1:tf.shape(gens)[1]-1]                                      # [bos]와 [eos] 제외한 gens.
# # target_idx = tf.where(tf.math.not_equal(gens2, get_token(token_dict, '[pad]')))
# # target_vals = tf.gather_nd(params = gens2, indices = target_idx)
# # candidates_maxlen = np.max(list(map(lambda x : len(x), candidates)))
# # gens3 = tf.scatter_nd(indices = target_idx, updates = target_vals, shape = (len(candidates), candidates_maxlen))        # 길이를 줄인 gens2.


# # hf_encoded_seqs = tf.cast(tf.zeros(shape = (0, candidates_maxlen)), dtype = tf.int32)
# repls = (' .', '.'), ('it \'s', 'it\'s'), ('i \'m', 'i\'m')
# for i in range(len(candidates)):
# # for i in range(tf.shape(gens3)[0]):
#     # a_decoded_seq = np.array(list(token_dict.values()))[gens3.numpy()[i, :]]
#     a_decoded_seq = np.array(list(token_dict.values()))[gens3.numpy()[i, :]]
#     print(' i : {}'.format(i))
#     decoded_str = ' '.join(a_decoded_seq)
#     decoded_str = reduce(lambda decoded_str, key_value: decoded_str.replace(*key_value), repls, decoded_str)
#     hf_encoded_seq = hf_gpt2_tokenizer(decoded_str, return_tensors='tf')['input_ids']
#     hf_encoded_seqs = tf.concat([hf_encoded_seqs, hf_encoded_seq], axis = 0)


# result_list = []
# for a_seq in full_texts:
#     result = ' '.join([t for t in a_seq])
#     result_list.append(result)

# hf_gpt2_tokenizer

# for t in a_decoded_seq:
#     print('t : ', t)
#     print(get_token(hf_token_dict, t))

# # '''
# # # PPL 점수 구하기
# # '''
# # # dec_outputs --> 사전훈련된 GPT 모델을 써서 뽑아야 하는듯 ... 근데 그러면 이 경우 drug-discovery에는 활용 안됨.
# # per_token_probs = tf.math.reduce_max(tf.nn.softmax(dec_outputs, axis = -1), axis = -1)
# # target_idx = tf.where(tf.math.equal(gens, get_token(token_dict, '[pad]')))
# # target_vals = tf.ones(shape = tf.shape(target_idx)[0])
# # per_token_probs_np = tf.tensor_scatter_nd_update(tensor = per_token_probs, indices = target_idx, updates = target_vals).numpy()

# # target_idx_rev = tf.where(tf.math.not_equal(gens, get_token(token_dict, '[pad]')))
# # target_idx_rev_pd = pd.DataFrame(target_idx_rev.numpy(), columns=['row_idx', 'col_idx'])
# # seq_len_wo_pad = np.array(target_idx_rev_pd.groupby('row_idx').size())
# # # ppl_score = tf.math.reduce_prod(per_token_probs_np, axis = -1) ** (-1/tf.shape(per_token_probs_np)[1].numpy())
# # ppl_score = tf.math.reduce_prod(per_token_probs_np, axis = -1) ** tf.cast(-1/seq_len_wo_pad, dtype = tf.float32)
# # print('ppl_min : {}, ppl_mean : {}, ppl_max : {}'.format(np.min(ppl_score), np.mean(ppl_score), np.max(ppl_score)))



# # %%
# '''
# baselines 평가
# '''
# os.chdir('/home/messy92/Leo/NAS_folder/ICML23/proposed')
# # transfer_direction = 'neg_to_pos'
# for baseline_name in ['CA', 'DL', 'LatentEdit', 'UNMT', 'CycleRL', 'DIRR-CA', 'DIRR-CO', 'DualRL', 'B-GST', 'G-GST', 'DO', 'RO', 'DR', 'TB', 'mult_dis', 'cond_dis', 'T&G']:
#     # baseline_name = 'CA'
#     transfer_direction = None
#     if transfer_direction == None:
        
#         baseline_gens_txt_all = []
#         for idx, transfer_direction in enumerate(['neg_to_pos', 'pos_to_neg']):
#             target_file_name = transfer_direction + ' (' + baseline_name + ').txt'
#             target_file_dir = find_all(target_file_name, os.getcwd())

#             # '''
#             # baseline들의 outputs.txt를 불러오는데, 이 때 텍스트 인코딩 문제로 "``" 이것이 "\'\'" 이렇게 표현된 경우가 있음. 이들을 수정해주어야 함.
#             # 마찬가지로 <UNK>, unk도 없애주어야 함.
#             # '''
#             # baseline_gens_txt = [line.replace("\'\'", "``").replace("<UNK>", "").replace("unk", "").split() for line in open(target_file_dir[0])] 
#             baseline_gens_txt = [line.split() for line in open(target_file_dir[0])] 

#             if idx == 0:
#                 baseline_gens_txt_all = baseline_gens_txt
#             else:
#                 baseline_gens_txt_all += baseline_gens_txt
        
#     else:
#         target_file_name = transfer_direction + ' (' + baseline_name + ').txt'
#         target_file_dir = find_all(target_file_name, os.getcwd())

#         baseline_gens_txt = [line.split() for line in open(target_file_dir[0])] 

#     # generated_inputs = [" ".join(a_gen) for a_gen in candidates]
#     generated_inputs = [" ".join(a_gen) for a_gen in baseline_gens_txt_all]

#     from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
#     checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     # raw_inputs = [
#     #     "I've been waiting for a HuggingFace course my whole life.",
#     #     "I hate this so much!",
#     #     "ever since joes has changed hands it 's just the best and freshest ."
#     # ]
#     generated_inputs = tokenizer(generated_inputs, padding=True, truncation=True, return_tensors="tf")
#     checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
#     model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
#     outputs = model(generated_inputs, training = False)
#     preds_class = tf.argmax(outputs.logits, axis = -1)

#     neg_to_pos_score = tf.reduce_sum(preds_class[:500])/500
#     pos_to_neg_score = tf.reduce_sum(preds_class[500:])/500
#     mean_score = (neg_to_pos_score + pos_to_neg_score)/2
#     print('baseline_name : {}, score : {}'.format(baseline_name, mean_score.numpy()))