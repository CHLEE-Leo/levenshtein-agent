# %%
import os
os.chdir('/home/messy92/Leo/NAS_folder/ICML23')
from utils_for_preprocess import *
import nltk
nltk.download('popular')
from nltk.tokenize import word_tokenize
import copy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_datasets as tfds
import pandas as pd
import csv
import numpy as np
import json
import pickle
import time
import datetime

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import argparse
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

def get_data(list_of_splits, attribute):
    '''
    split : train, test, dev, reference ...
    attribute : 0, 1 --> negative, positive
    '''

    target_data_list_dir = []    
    for split in list_of_splits:
        data_dir = './data/text-style-transfer/Li et al/' + 'yelp'

        data_list_dir = os.listdir(data_dir)
        if attribute == None:
            target_data_list = [data_split for data_split in data_list_dir if str(split) in data_split.split('.')]
        else:
            target_data_list = [data_split for data_split in data_list_dir if str(split) in data_split.split('.') and str(attribute) in data_split.split('.')]
        target_data_list_dir += [data_dir + '/' + src for src in target_data_list]
        
    src_lines = []
    src_lines += [word_tokenize(l.lower()) for data_split_dir in target_data_list_dir for l in open(data_split_dir, 'r')]
    src_lines = [['[BOS]'] + x + ['[EOS]'] for x in src_lines]

    return src_lines

if args.data == 'yelp':

    text_path = "./prep_data/text-style-transfer/txt_format"

    # vocabulary 생성
    full_texts = get_data(['train', 'test', 'dev'], None)
    tokenizer = Tokenizer(oov_token = "[unk]")
    tokenizer.fit_on_texts(full_texts)
    full_encoded_sequences = tokenizer.texts_to_sequences(full_texts)
    full_encoded_sequences = pad_sequences(full_encoded_sequences, padding='post')
    print('full_set :', full_encoded_sequences.shape)

    vocab_dict = tokenizer.word_index
    # a = {'[pad]' : 0}
    # a.update(vocab_dict)
    # vocab_dict = copy.deepcopy(a)
    vocab_dict = copy.deepcopy(dict(map(reversed, vocab_dict.items())))

    ## pickle 포맷 저장
    with open('./prep_data/text-style-transfer' + '/token_dict.pickle', 'wb') as f:
        pickle.dump(vocab_dict, f)

    # train 데이터 셋 생성
    neg_texts = get_data(['train'], 0)
    neg_polarity = [0] * len(neg_texts)
    pos_texts = get_data(['train'], 1)
    pos_polarity = [1] * len(pos_texts)

    train_texts = neg_texts + pos_texts
    train_polarity = neg_polarity + pos_polarity

    train_encoded_sequences = tokenizer.texts_to_sequences(train_texts)
    train_encoded_sequences = pad_sequences(train_encoded_sequences, padding='post')
    print('train_set :', train_encoded_sequences.shape)

    np.save('./prep_data/text-style-transfer' + '/input_sequence(train).npy', train_encoded_sequences)
    np.save('./prep_data/text-style-transfer' + '/attribute(train).npy', train_polarity)

    # val 데이터 셋 생성
    neg_texts = get_data(['dev'], 0)
    neg_polarity = [0] * len(neg_texts)
    pos_texts = get_data(['dev'], 1)
    pos_polarity = [1] * len(pos_texts)

    total_texts = neg_texts + pos_texts
    val_polarity = neg_polarity + pos_polarity

    val_encoded_sequences = tokenizer.texts_to_sequences(total_texts)
    val_encoded_sequences = pad_sequences(val_encoded_sequences, padding='post', maxlen = train_encoded_sequences.shape[1])
    print('val_set :', val_encoded_sequences.shape)

    np.save('./prep_data/text-style-transfer' + '/input_sequence(val).npy', val_encoded_sequences)
    np.save('./prep_data/text-style-transfer' + '/attribute(val).npy', val_polarity)


    # test 데이터 셋 생성
    neg_texts = get_data(['test'], 0)
    neg_polarity = [0] * len(neg_texts)
    pos_texts = get_data(['test'], 1)
    pos_polarity = [1] * len(pos_texts)

    total_texts = neg_texts + pos_texts
    test_polarity = neg_polarity + pos_polarity

    test_encoded_sequences = tokenizer.texts_to_sequences(total_texts)
    test_encoded_sequences = pad_sequences(test_encoded_sequences, padding='post', maxlen = train_encoded_sequences.shape[1])
    print('test_set :', test_encoded_sequences.shape)

    np.save('./prep_data/text-style-transfer' + '/input_sequence(test).npy', test_encoded_sequences)
    np.save('./prep_data/text-style-transfer' + '/attribute(test).npy', test_polarity)

elif args.data == 'drug_repurposing_hub':
    text_path = "./prep_data/drug-discovery/txt_format"

    '''
    drug_repurposing data
    '''
    # # 화합물 샘플 데이터
    # sample_dat = pd.read_csv('./data/drug-discovery/drug_repurposing_hub/repurposing_samples_20200324.csv')     # 13553 x 12
    # sample_dat_unique = sample_dat.drop_duplicates(subset='smiles', keep = 'first')                             # 6806 x 12

    # # 약물 메타정보 데이터
    # drug_dat = pd.read_csv('./data/drug-discovery/drug_repurposing_hub/repurposing_drugs_20200324.csv')         # 6798 x 6

    # # 화합물 샘플 x 약물 메타정보 데이터
    # sample_x_drug_dat = pd.merge(sample_dat_unique, drug_dat, on = 'pert_iname', how = 'outer')                 # 6798 x 17

    # # 실제 론칭된 약물만 필터링
    # launched_sample_x_drug_dat = sample_x_drug_dat[sample_x_drug_dat['clinical_phase'] == 'Launched']           # 2427 x 17

    # # drug repurposing 성공 케이스가 10개 이상인 질병 필터링
    # per_disease_count = launched_sample_x_drug_dat['disease_area'].value_counts()
    # N = 10
    # target_idx = np.where(per_disease_count.values >= N)[0]
    # top_N_disease_name = per_disease_count.index[target_idx]
    # top_N_disease_freq = per_disease_count.values[target_idx]

    # # 질병 별 화합물 갯수 보기
    # plt.style.use('ggplot')
    # fig, ax = plt.subplots()
    # bars = ax.barh(top_N_disease_name, top_N_disease_freq)

    # ax.bar_label(bars)

    # for bars in ax.containers:
    #     ax.bar_label(bars)

    '''
    gyumin data
    '''
    # 화합물 샘플 데이터
    sample_dat = pd.read_csv('./data/drug-discovery/gyumin/BindingDB_repositioned.csv')     # 51834 x 6

    # 약물 메타정보 데이터
    drug_dat = pd.read_csv('./data/drug-discovery/gyumin/Repositioned_edited.csv')     # 41 x 7

    # 화합물 샘플 x 약물 메타정보 데이터
    # --> 타겟 단백질 시퀀스 (= 'Target_seq' = 'Sequence')에 작용하는 모든 화합물 샘플들 구축
    sample_x_drug_dat = pd.merge(sample_dat, drug_dat, left_on = 'Target_seq', right_on = 'Sequence', how = 'inner')     # 27667 x 13

    # 질병 컬럼 범주화
    sample_x_drug_dat['Related disease'] = sample_x_drug_dat['Related disease'].astype('category')
    sample_x_drug_dat['Disease_code'] = sample_x_drug_dat['Related disease'].cat.codes

    # 질병 코드맵 저장
    disease_code_map = dict(enumerate(sample_x_drug_dat['Related disease'].cat.categories))
    with open('./prep_data/drug-discovery/disease_code_map.pickle', 'wb') as f:
        pickle.dump(disease_code_map, f)

    # 훈련용 + 검증용 데이터 셋
    train_val_dat = sample_x_drug_dat[['Compound_smiles', 'Sequence', 'Compound_len', 'Target_len', 'Related drug', 'Related disease', 'Disease_code']]     # 27667 x 6

    # 평가용 데이터 셋
    test_dat = sample_x_drug_dat[['Related drug', 'Related disease', 'Indication', 'Sequence', 'Compound_smiles', 'Disease_code']]  # 27667 x 5

    # tokenizer를 위해 전체 데이터 (훈련용-검증용 구분 X)를 .txt 파일로 저장후 다시 불러오기
    total_sample_dat = pd.concat([sample_x_drug_dat[['Compound_smiles']].rename(columns={'Compound_smiles':'sample'}), sample_x_drug_dat[['Target_seq']].rename(columns={'Target_seq':'sample'})], axis = 0)
    total_sample_dat_for_tokenizing = total_sample_dat['sample'].apply(lambda x: ['[BOS]'] + list(x) + ['[EOS]'])

    # vocabulary 생성
    tokenizer = Tokenizer(oov_token = "[unk]", filters='!"%&*,/;?^_`{|}~\t\n')
    tokenizer.fit_on_texts(total_sample_dat_for_tokenizing)
    full_encoded_sequences = tokenizer.texts_to_sequences(total_sample_dat_for_tokenizing)
    full_encoded_sequences = pad_sequences(full_encoded_sequences, padding='post')
    print('full_set :', full_encoded_sequences.shape)

    vocab_dict = tokenizer.word_index
    # a = {'[pad]' : 0}
    # a.update(vocab_dict)
    # vocab_dict = copy.deepcopy(a)
    vocab_dict = copy.deepcopy(dict(map(reversed, vocab_dict.items())))

    # pickle 포맷 dictionary 저장
    with open('./prep_data/drug-discovery' + '/token_dict.pickle', 'wb') as f:
        pickle.dump(vocab_dict, f)

    # pickel 포맷 tokenizer 저장
    with open('./prep_data/drug-discovery/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    train = 훈련
    val = 검증
    test = 평가
    train_texts_input = 타겟 단백질 RNA 시퀀스
    train_texts_output = 화합물 시퀀스
    '''

    # train_dat을 다시 train_dat - val_dat으로 나누기
    train_idx = train_val_dat.sample(frac = 0.8, random_state = 920807).index
    val_bool = np.invert(train_val_dat.index.isin(list(train_idx)))
    val_idx = list(train_val_dat.index[val_bool])    
    train_dat = train_val_dat.iloc(axis = 0)[train_idx]
    val_dat = train_val_dat.iloc(axis = 0)[val_idx]
    train_dat.to_csv('./prep_data/drug-discovery' + '/train_dat.csv')

    # train 데이터 셋 생성
    train_texts_input = train_dat['Sequence'].apply(lambda x : ['[bos]'] + list(x) + ['[eos]'])
    train_texts_output = train_dat['Compound_smiles'].apply(lambda x : ['[bos]'] + list(x) + ['[eos]'])
    train_polarity = train_dat['Disease_code']

    train_encoded_sequences_input = tokenizer.texts_to_sequences(train_texts_input)
    train_encoded_sequences_input = pad_sequences(train_encoded_sequences_input, padding='post')
    print('train_input_set :', train_encoded_sequences_input.shape)

    train_encoded_sequences_output = tokenizer.texts_to_sequences(train_texts_output)
    train_encoded_sequences_output = pad_sequences(train_encoded_sequences_output, padding='post')
    print('train_output_set :', train_encoded_sequences_output.shape)

    np.save('./prep_data/drug-discovery' + '/input_sequence(train).npy', train_encoded_sequences_input)
    np.save('./prep_data/drug-discovery' + '/output_sequence(train).npy', train_encoded_sequences_output)
    np.save('./prep_data/drug-discovery' + '/attribute(train).npy', train_polarity)


    # val 데이터 셋 생성
    val_texts_input = val_dat['Sequence'].apply(lambda x : ['[bos]'] + list(x) + ['[eos]'])
    val_texts_output = val_dat['Compound_smiles'].apply(lambda x : ['[bos]'] + list(x) + ['[eos]'])
    val_polarity = val_dat['Disease_code']

    val_encoded_sequences_input = tokenizer.texts_to_sequences(val_texts_input)
    val_encoded_sequences_input = pad_sequences(val_encoded_sequences_input, padding='post')
    print('val_set :', val_encoded_sequences_input.shape)

    val_encoded_sequences_output = tokenizer.texts_to_sequences(val_texts_output)
    val_encoded_sequences_output = pad_sequences(val_encoded_sequences_output, padding='post')
    print('val_set :', val_encoded_sequences_output.shape)

    np.save('./prep_data/drug-discovery' + '/input_sequence(val).npy', val_encoded_sequences_input)
    np.save('./prep_data/drug-discovery' + '/output_sequence(val).npy', val_encoded_sequences_output)
    np.save('./prep_data/drug-discovery' + '/attribute(val).npy', val_polarity)

    # test 데이터 셋 생성
    test_texts_input = test_dat['Sequence'].apply(lambda x : ['[bos]'] + list(x) + ['[eos]'])
    test_texts_output = test_dat['Compound_smiles'].apply(lambda x : ['[bos]'] + list(x) + ['[eos]'])
    test_polarity = test_dat['Disease_code']

    test_encoded_sequences_input = tokenizer.texts_to_sequences(test_texts_input)
    test_encoded_sequences_input = pad_sequences(test_encoded_sequences_input, padding='post')
    print('test_set :', test_encoded_sequences_input.shape)

    test_encoded_sequences_output = tokenizer.texts_to_sequences(test_texts_output)
    test_encoded_sequences_output = pad_sequences(test_encoded_sequences_output, padding='post')
    print('test_set :', test_encoded_sequences_output.shape)

    np.save('./prep_data/drug-discovery' + '/input_sequence(test).npy', test_encoded_sequences_input)
    np.save('./prep_data/drug-discovery' + '/output_sequence(test).npy', test_encoded_sequences_output)
    np.save('./prep_data/drug-discovery' + '/attribute(test).npy', test_polarity)


    '''
    # -------------------------------------------------------------------------------------------------------------------------
    '''

    # 질병 별 화합물 갯수 보기
    per_disease_count = train_val_dat['Related disease'].value_counts()
    N = 1
    target_idx = np.where(per_disease_count.values >= N)[0]
    top_N_disease_name = per_disease_count.index[target_idx]
    top_N_disease_freq = per_disease_count.values[target_idx]


    tick_label_list = ['Stomatognathic\n diseases', 'Congenital, hereditary\n and neonatal diseases', 
                        'Nervous system\n diseases', 'Neoplasm', 
                        'Bacterial infection\n and mycoses', 'Parasitic\n diseases']
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    bars = ax.bar(top_N_disease_name, top_N_disease_freq, 
                    tick_label=tick_label_list)

    ax.bar_label(bars)
    ax.set_xticklabels(tick_label_list, rotation = 45, ha="right")
    
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.savefig('drug-reposition-distribution.pdf', bbox_inches = 'tight')

    train_val_dat.groupby(['Related disease']).mean()
    train_val_dat.groupby(['Related disease']).count()
    drug_dat.groupby(['Related disease', 'Indication']).count()
    drug_dat.groupby(['Related disease', 'Indication']).mean()

    


# %%
