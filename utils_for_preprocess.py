# %%
'''
This file contains the utilities (e.g., functions, classes) that is able to import and to use in any files including proprocessing.py and model.py.
'''
import os
import copy
from re import split
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from itertools import islice

'''
-- (1) Preprocessing 용 utils
'''
# 데이터 로더
def data_loader(data_address):
  use_cols = ['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
  sample_data = pd.read_csv(data_address, usecols = use_cols) # use_cols에 해당하는 변수들만 read 하기
  sample_data = sample_data[['BindingDB Target Chain Sequence', 'Ligand SMILES', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']] # 변수들의 순서 바꿔주기; 0번째 컬럼 = A_Partition, 1번째 컬럼 = B_Partition.
  return sample_data

# 시퀀스 길이 기반 데이터 절삭기
'''
truncate_rate (절삭률) : 시퀀스 길이 상위 truncate_rate (%) 파라미터; 
시퀀스 길이 상위 truncate_rate (%) 이상인 시퀀스들은 절삭 (제거)
'''
def len_based_truncation(sample_data, truncate_rate = 0.2):

  # 데이터 샘플링
  # (1) 단백질과 화합물의 시퀀스 길이 분포 추출
  protein_len_distribution = sample_data.apply(lambda x : len(x['BindingDB Target Chain Sequence']), axis = 1)
  compound_len_distribution = sample_data.apply(lambda x : len(x['Ligand SMILES']), axis = 1)

  # (2-1) 단백질의 경우 소수의 너무 긴 단백질이 존재하므로 길이 하위 1 - truncate_rate (%) 까지만 고려. 즉 시퀀스 길이 상위 truncate_rate (%)의 단백질은 제거
  protein_quantile = protein_len_distribution.quantile(1 - truncate_rate)
  truncated_protein_len_distribution = protein_len_distribution[protein_len_distribution.lt(protein_quantile)]

  # (2-2) protein sequence는 길이 하위 1 - truncate_rate (%) (or 길이 상위 truncate_rate (%))에 해당하는 샘플만 고려하는 truncated_data 만들어주기
  truncated_data = sample_data[protein_len_distribution.lt(protein_quantile)]

  return truncated_data, protein_len_distribution, truncated_protein_len_distribution, compound_len_distribution

# 텍스트 클린징
def clean_text(review):
    cleaned = review.replace("\\n", " ")
    cleaned = cleaned.replace("\'", "'")
    cleaned = cleaned.replace("\\r", " ")
    cleaned = cleaned.replace("\\""", " ")
    cleaned = cleaned.replace("  ", " ")
    return cleaned

# 시퀀스 토크나이저
'''
데이터 내 시퀀스의 토큰들을 추출, 정수 임베딩, 사전생성을 수행
'''
# gpt는 input_sequence와 target_sequence의 domain이 공유된 데이터를 다룸.
# 즉, input_sequence와 target_sequence가 사실상 같은 sequence이므로 단일 시퀀스에 대해서만 tokenizer를 적용해주면 됨.
def data_tokenizer(truncated_data, dict_size):
  # 텐서플로 텍스트 및 시퀀스 라이브러리 설치
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tokenizers import BertWordPieceTokenizer

    tokenizer = BertWordPieceTokenizer(lowercase=False, trip_accents=False)

    # (0) <bos>와 <eos> 토큰 추가 및 시퀀스 정수 임베딩
    A_tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t', lower = True, split = ' ', num_words = dict_size)

    A_sequence = '<bos> ' + truncated_data.iloc[:, 0] + ' <eos>'
    A_tokenizer.fit_on_texts(A_sequence)
    encoded_sequence = A_tokenizer.texts_to_sequences(A_sequence)

    # (1) 시퀀스의 최대길이 (BOS, EOS 추가 후) 추출
    A_maxlen = A_sequence.apply(lambda x : len(x.split())).max()

    # (2) 시퀀스 패딩하기
    padded_sequence = pad_sequences(encoded_sequence, maxlen = A_maxlen, padding = 'post')

    # # (3) 저빈도 단어 제거
    # token_bincounts = np.bincount(padded_sequence.flatten())[1:]

    # (4) 사전 만들기
    A_dict = copy.deepcopy(A_tokenizer.word_index)
    A_dict['<pad>'] = 0
    A_dict = dict(sorted(A_dict.items(), key = lambda item : item[1]))
    A_dict_slice = dict(islice(A_dict.items(), dict_size))
    A_dict_reverse = dict(map(reversed, A_dict_slice.items()))

    # # (5) 사전-빈도 데이터프레임 만들기
    # token_freq_df = pd.DataFrame(token_bincounts)
    # token_freq_df.index = list(A_dict.keys())
    # token_freq_df.columns = ['freq']

    return padded_sequence, A_dict_reverse

def pair_instance_sampling(batch_polarity, batch_sequence, dataset_polarity, dataset_sequence):
    # 현 배치에서 polarity가 neg/pos인 샘플들의 index 추출
    batch_neg_polar_idx = np.where(batch_polarity == 0)[0]
    batch_pos_polar_idx = np.where(batch_polarity == 1)[0]
    num_batch_neg_polar = len(batch_neg_polar_idx)
    num_batch_pos_polar = len(batch_pos_polar_idx)

    # 전체 데이터 셋에서 polarity가 neg/pos인 샘플들의 index 추출
    dataset_neg_polar_idx = np.where(dataset_polarity == 0)[0]
    dataset_pos_polar_idx = np.where(dataset_polarity == 1)[0]

    # (전체 데이터 셋으로부터) 현 배치에 대한 positive-pair / negative-pair의 index 구하기
    batch_neg_pospair_idx = np.random.choice(dataset_neg_polar_idx, size = num_batch_neg_polar, replace = False) # neg 배치의 positive-pair : 전체 데이터셋에서 negative polarity인 neg 배치 크기 만큼의 샘플
    batch_neg_negpair_idx = np.random.choice(dataset_pos_polar_idx, size = num_batch_neg_polar, replace = False) # neg 배치의 negative-pair : 전체 데이터셋에서 positive polarity인 neg 배치 크기 만큼의 샘플
    batch_pos_pospair_idx = np.random.choice(dataset_pos_polar_idx, size = num_batch_pos_polar, replace = False) # pos 배치의 positive-pair : 전체 데이터셋에서 positive polarity인 pos 배치 크기 만큼의 샘플
    batch_pos_negpair_idx = np.random.choice(dataset_neg_polar_idx, size = num_batch_pos_polar, replace = False) # pos 배치의 negative-pair : 전체 데이터셋에서 negative polarity인 pos 배치 크기 만큼의 샘플

    # (전체 데이터 셋으로부터) 현 배치에 대한 positive-pair / negative-pair 시퀀스 가져오기
    batch_neg_pospair_seq = dataset_sequence[batch_neg_pospair_idx, :]
    batch_neg_negpair_seq = dataset_sequence[batch_neg_negpair_idx, :]
    batch_pos_pospair_seq = dataset_sequence[batch_pos_pospair_idx, :]
    batch_pos_negpair_seq = dataset_sequence[batch_pos_negpair_idx, :]

    # batch_pair 정의
    batch_pospair = copy.deepcopy(batch_sequence)
    batch_negpair = copy.deepcopy(batch_sequence)
    ## pos_pair
    batch_pospair[batch_neg_polar_idx, :] = batch_neg_pospair_seq
    batch_pospair[batch_pos_polar_idx, :] = batch_pos_pospair_seq
    ## neg_pair
    batch_negpair[batch_neg_polar_idx, :] = batch_neg_negpair_seq
    batch_negpair[batch_pos_polar_idx, :] = batch_pos_negpair_seq

    # # 검증 코드
    # print(tokenizer.decode(origin_batch[0, :50]))
    # print(tokenizer.decode(pos_pair_batch[0, :50]))
    # print(tokenizer.decode(neg_pair_batch[0, :50]))

    return batch_sequence, batch_pospair, batch_negpair

# # transformer는 input_sequence와 target_sequence의 domain이 분리되어 있는 데이터를 다룸.
# # 때문에, tokenizer도 gpt와 달리, 양쪽 시퀀스 모두에게 적용되어야 함.
# def data_tokenizer_for_transformer(truncated_data):
#   # 텐서플로 텍스트 및 시퀀스 라이브러리 설치
#   from tensorflow.keras.preprocessing.text import Tokenizer
#   from tensorflow.keras.preprocessing.sequence import pad_sequences

#   # (0) <bos>와 <eos> 토큰 추가
#   A_sequence = truncated_data.iloc[:, 0].apply(lambda x : ['<bos>'] + list(x) + ['<eos>'])
#   B_sequence = truncated_data.iloc[:, 1].apply(lambda x : ['<bos>'] + list(x) + ['<eos>'])

#   # (1) 시퀀스 정수 임베딩
#   ## A_Partition 내 시퀀스 (노드)들 토크나이징
#   A_tokenizer = Tokenizer(filters = ' ', lower = False)
#   A_tokenizer.fit_on_texts(A_sequence)
#   encoded_A_sequence = A_tokenizer.texts_to_sequences(A_sequence)

#   ## B_Partition 내 시퀀스 (노드)들 토크나이징
#   B_tokenizer = Tokenizer(filters = ' ', lower = False)
#   B_tokenizer.fit_on_texts(B_sequence)
#   encoded_B_sequence = B_tokenizer.texts_to_sequences(B_sequence)

#   # (2) 시퀀스의 최대길이 (BOS, EOS 추가 후) 추출
#   A_maxlen = A_sequence.apply(lambda x : len(x)).max()
#   B_maxlen = B_sequence.apply(lambda x : len(x)).max()

#   # (3) 시퀀스 패딩하기
#   padded_A_sequence = pad_sequences(encoded_A_sequence, maxlen = A_maxlen, padding = 'post')
#   padded_B_sequence = pad_sequences(encoded_B_sequence, maxlen = B_maxlen, padding = 'post')

#   # (4) 사전 만들기
#   A_dict = copy.deepcopy(A_tokenizer.word_index)
#   A_dict_reverse = dict(map(reversed, A_dict.items()))
#   B_dict = copy.deepcopy(B_tokenizer.word_index)
#   B_dict_reverse = dict(map(reversed, B_dict.items()))

#   return padded_A_sequence, padded_B_sequence, A_dict_reverse, B_dict_reverse

# 각 sequence의 token을 -> word로 변환하는 함수 (from sequence to sentence)
def sequence_to_sentence(a_sequence, target_token_dict):
    # Get split list of keys and values.
    keys, vals = zip(*target_token_dict.items())

    # if (0 in keys) == False:
    #     keys = np.append(0, keys)
    #     vals = np.append('<pad>', vals)
    # else:
    #     keys = np.array(keys)
    #     vals = np.array(vals)

    keys = np.array(keys)
    vals = np.array(vals)

    token_idx_in_dict = [np.where(keys == token)[0][0] for token in a_sequence]
    a_sentence = [list(vals)[token_idx] for token_idx in token_idx_in_dict]
    return a_sentence

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def tsne_embedding(word_embedding_mat, vocab_size = 500, n_dimensions = 2):
    
    model = TSNE(n_components = n_dimensions)
    tsne_embedding = model.fit_transform(word_embedding_mat[:vocab_size, :])

    return tsne_embedding

def visualizing_tsne(tsne_embedding, figure_name, figure_type, token_dict, vocab_size):

    vocab_show = list(token_dict.values())[:vocab_size]
    embedding_show = tsne_embedding[:vocab_size, :]
    tsne_df = pd.DataFrame(embedding_show, index = vocab_show, columns=['x', 'y']) 

    fig = plt.figure() 
    fig.set_size_inches(13, 13) 
    ax = fig.add_subplot(1, 1, 1) 
    ax.scatter(tsne_df['x'], tsne_df['y']) 
    for word, pos in tsne_df.iterrows(): 
        ax.annotate(word, pos, fontsize=4)

    plt.title('Word ' + figure_type + ' Embedding')
    plt.xlabel("x-axis") 
    plt.ylabel("y-axis") 
    plt.savefig('/home/messy92/Leo/NAS_folder/ICML22/results/text-style-transfer/' + figure_name + '.png', dpi = 300)

    return None

'''
transform str to bool.
'''
def str2bool(v): 
  if isinstance(v, bool): 
      return v 
  if v.lower() in ('yes', 'true', 't', 'y', '1'): 
      return True 
  elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
      return False 
  else: 
      raise argparse.ArgumentTypeError('Boolean value expected.')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

'''
preprocessor class that deals with nutrition data.
'''
class nutrition_preprocessor():
  def __init__(self, **preprocessing_kwargs):
      self.feature_data = preprocessing_kwargs['feature_data']

  # insert the features of padding token into nutrition data
  # types of padding token : {empty, bos, eos}
  def insert_padding_feature(self):
      
      # empty token feature
      empty_vector = pd.DataFrame(0, columns = self.feature_data.columns, index = [0])
      empty_vector['name'] = "empty"
      empty_vector['Class'] = "빈칸"
      self.feature_data = pd.concat([empty_vector, self.feature_data]).reset_index(drop = True)

      # bos token feature
      start_vector = pd.DataFrame(0, columns = self.feature_data.columns, index = [0])
      start_vector['name'] = "시작"
      start_vector['Class'] = "앞뒤"
      self.feature_data = pd.concat([self.feature_data, start_vector]).reset_index(drop = True)
  
      # eos token feature
      end_vector = pd.DataFrame(0, columns = self.feature_data.columns, index = [0])
      end_vector['name'] = "종료"
      end_vector['Class'] = "앞뒤"
      self.feature_data = pd.concat([self.feature_data, end_vector]).reset_index(drop = True)

      return self.feature_data

  # Get nutrient-related features from feature data
  def get_nutrient_features(self, feature_data):

      # Call the name of features and use them only to make nutrition data.
      nutrient_feature = list(feature_data.columns.values)
      # nutrient_feature = [e for e in nutrient_feature if e not in ["Weight", "Class", "dish_class1", "dish_class2", "meal_class"]]
      nutrient_feature = [e for e in nutrient_feature if e not in ["Class"]]
      nutrient_feature_data = feature_data.loc(axis = 1)[nutrient_feature]
      # nutrient_feature_data['name'] = nutrient_feature_data['name'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
      nutrient_feature_data['name'] = nutrient_feature_data['name'].str.replace(" ", "")

      return nutrient_feature_data

  def __call__(self):

      # Make new feature dataset where the padding tokens and their corresponding values are inserted.
      new_feature_data = self.insert_padding_feature()

      # Split nutrient feature vector from and make nutrition data
      nutrient_data = self.get_nutrient_features(new_feature_data)

      ## 메뉴 dictionary
      food_dict = dict(new_feature_data['name'])

      return nutrient_data, food_dict
'''
preprocessor class that deals with diet data.
'''
class diet_sequence_preprocessor():
  def __init__(self, **preprocessing_kwargs):

      # Define global varaibles
      self.diet_data = preprocessing_kwargs['sequence_data']
      self.quality = preprocessing_kwargs['DB_quality']
      self.integrate_DB = preprocessing_kwargs['integrate']

      # Define diet indices initialized by OR, arranged and corrected by expert
      self.or_idx = np.array(range(0, self.diet_data.shape[0], 4))        # diet initialized by OR.
      self.expert_idx1 = np.array(range(1, self.diet_data.shape[0], 4))   # diet arranged by expert.
      self.expert_idx2 = np.array(range(2, self.diet_data.shape[0], 4))   # diet corrected by expert with spec-checker.
      self.expert_idx3 = np.array(range(3, self.diet_data.shape[0], 4))   # diet corrected by expert without spec-checker.

  def select_DB_quality(self):

      if self.quality == 'or':
          diet_data = self.diet_data.iloc[self.or_idx]

      elif self.quality == 'arrange':
          diet_data = self.diet_data.iloc[self.expert_idx1]

      elif self.quality == 'correct1':
          diet_data = self.diet_data.iloc[self.expert_idx2]

      elif self.quality == 'correct2':
          diet_data = self.diet_data.iloc[self.expert_idx3]
          
      return diet_data

  def check_by_nutrient_data(self, diet_data, nutrient_data):

      # Replace possible typo with blank and fill "empty" into the elements that has NaN value.
      # diet_data = diet_data.replace('[^\w]', '', regex=True)
      diet_data.fillna("empty", inplace = True)

      # Get the set of menus used in diet_data
      menus_in_dietDB = set(np.unique( np.char.strip(diet_data.values.flatten().astype('str')) ))                                  
      # Get the set of menus recorded in nutrient_data
      menus_in_nutritionDB = set(np.unique( np.char.strip(nutrient_data['name'].values.flatten().astype('str'))))                  

      # Get the menus which are used in diet data but not exist in nutrient data
      menus_only_in_dietDB = menus_in_dietDB.difference(menus_in_nutritionDB)                                                      

      # print('Total {} menus exist in \'{}\' diet_data'.format(len(menus_in_dietDB), self.quality))
      # print('Total {} menus exist in nutrition_data (menu data)'.format(len(menus_in_nutritionDB)))
      # print('There were {} mismatched menus between \'{}\' diet_data and nutrition_data (menu_data)'.format(len(menus_only_in_dietDB), self.quality))

      if len(menus_only_in_dietDB) > 0:
          # Store the menus that do not exist in nutrient data
          pd.DataFrame(menus_only_in_dietDB).to_csv('./results/diet-planning/menus_only_in_dietDB.csv', encoding="utf-8-sig")

          # Replace the values with 'empty', that exist in diet_data but not in nutrient_data.
          empty_filled_diet_data = diet_data.replace(menus_only_in_dietDB, 'empty')

          return empty_filled_diet_data
      else:
          return diet_data

  def __call__(self, nutrient_data):

      # if an input diet data consists of multiple source of generation
      if self.integrate_DB == True:
          # Get diet data according to quality which is given by user-defined parameter.
          diet_data = self.select_DB_quality()

      # if an input diet data has a signel source of generation
      else:
          diet_data = self.diet_data

      # Make padding whose the types of padding tokens include 'bos', 'eos', and 'empty', and insert into diet data.
      diet_data.insert(loc = 0, column = "Start", value = ["시작"] * diet_data.shape[0])
      diet_data.insert(loc = diet_data.shape[1], column = "End", value = ["종료"] * diet_data.shape[0])

      # Cross check the mismatched menus using nutrient_data
      diet_data = self.check_by_nutrient_data(diet_data, nutrient_data)

      return diet_data

# Mapping diet data to incidence data
def diet_to_incidence(diet_data_np, food_dict):
  incidence_mat = tf.zeros([len(food_dict), diet_data_np.shape[1]])

  for i in range(diet_data_np.shape[0]):
      incidence_mat += tf.transpose(tf.one_hot(diet_data_np[i, :], depth = len(food_dict)))

  return incidence_mat
    
# Mapping food to token
def food_to_token(diet_data, nutrient_data, empty_delete = False, num_empty = 2):
  from tqdm import tqdm

  '''
  empty_delete : empty 갯수 지정해서 지정한 갯수 이상 포함된 식단ㅇ들 제거할지 여부. default = False이고, 이 경우 empty로 꽉찬 행만 제거
  num_empty : empty 갯수
  '''

  diet_data_np = np.zeros([diet_data.shape[0], diet_data.shape[1]])
  # Mapping from food to token
  ## 영양소 data기준으로 식단 data tokenization
  delete_list = np.array([])
  for i in tqdm(range(diet_data.shape[0])):

      empty_list = np.array([])

      for j in range(diet_data.shape[1]):
          try:
              # tokenization
              diet_data_np[i, j] = nutrient_data[nutrient_data['name'] == diet_data.iloc[i, j]].index[0]

              # 각 식단마다 등장한 empty의 갯수를 담은 리스트 생성
              if diet_data_np[i, j] == 0:
                  empty_list = np.append(empty_list, j)
                  # print('i :{}, j : {}, empty_list : {}'.format(i, j, empty_list))
          except:
              pass

      # print('diet : {}'.format(sequence_to_sentence([diet_data_np[i, :]], food_dict)))

      # i번쨰 식단에 등장한 empty의 갯수가 num_empty보다 클 경우 해당 식단 i를 삭제해야할 대상인 delete_list에 담기
      if len(empty_list) > num_empty:
          delete_list = np.append(delete_list, i)

  print(' ')
  print('{} diets are deleted as they have more than {} empty slots'.format(len(delete_list), num_empty))

  # Get indices of diets that the number of empty is larger than num_empty.
  if empty_delete == True:
      non_value_idx = copy.deepcopy(delete_list)
                                                  

  # Delete indices of diets that the number of empty is larger than num_empty.
  diet_data_np = np.delete(diet_data_np, non_value_idx.astype(int), axis = 0) 

  return tf.cast(diet_data_np, dtype = tf.int32)
# %%
dir = '/home/messy92/Leo/Controlled_Sequence_Generation/Latent-Control/data/story-telling/writingPrompts/'
def load_dataset(dir):
    data = ["train", "test", "valid"]
    for name in data:
        with open(dir + name + ".wp_target") as f:
            stories = f.readlines()
        stories = [" ".join(i.split()[0:1000]) for i in stories]
        with open(dir + name + ".wp_target", "w") as o:
            for line in stories:
                o.write(line.strip() + "\n")

# p-벡터 샘플링 (디리클레 분포 기반) 함수
def dirichlet_sampling(input_sample, alpha_vector, sample_wise = False):
    # alpha_vector : 디리클레 파라미터 (e.g., 균일확률분포 = [1, 1, 1], 불균일 확률분포 = [4, 2, 1])
    # sample_wise : sample 별로 서로 다른 probability를 주려면 (= sample_wise 디리클레 샘플링 하라면) True, 그게 아니면 False.

    num_samples = input_sample.shape[0]

    if sample_wise == True:
      p_vec = np.random.dirichlet(alpha_vector, size = num_samples)
    else:
      p_vec = np.random.dirichlet(alpha_vector, size = 1)
      p_vec = np.tile(p_vec, reps = (num_samples, 1))

    return p_vec

# 잠재 공간 심플렉스 연산
def simplex_propagation(p_vec, z_mat):
  # z_mat : simplex 적용되기 전 layer의 반환 값 (잠재행렬)

  per_p_latent_dim = z_mat.shape[1] // p_vec.shape[1]
  p_mat = np.empty((z_mat.shape[0], 0))

  for i in range(p_vec.shape[1]):
      p_partial_mat = np.tile(p_vec[:, i], (per_p_latent_dim, 1)).transpose()        
      p_mat = np.append(p_mat, p_partial_mat, axis = 1)
    
  simplex_mat = z_mat * p_mat
  return simplex_mat

# 현재 데이터 (input)의 positive & negative pair 샘플링하는 함수
## (1) Text Style Transfer (TST) samplar
### 특정 수준의 값 (e.g., 1 또는 0 이길 원함)
class TST_Samplar:
    def __init__(self, seq_dataset, table_dataset):
        self.polarity = table_dataset['label']      # total sequences
        self.sequence = seq_dataset[0]

    # Sample a positive pair of input (except for input itself)
    def pos_samplar(self, input_sample):
      size_of_batch = input_sample.shape[0]           # get the size of batch
      pos_polars = self.polarity[self.polarity == 1]  # get the positive indices from training set     
      rand_sample = np.random.choice(pos_polars.index, size = size_of_batch)  # random sampling of positive samples 

      return self.sequence[rand_sample, :], rand_sample

    # Sample a negative pair of input (except for input itself)
    def neg_samplar(self, input_sample):
      size_of_batch = input_sample.shape[0]           # get the size of batch
      neg_polars = self.polarity[self.polarity == 0]  # get the negative indices from training set
      rand_sample = np.random.choice(neg_polars.index, size = size_of_batch) # random sampling of negative samples w batch size

      return self.sequence[rand_sample, :], rand_sample

    def __call__(self, input_sample):
      pos_samples, _ = self.pos_samplar(input_sample)
      neg_samples, _ = self.neg_samplar(input_sample)

      return (pos_samples, neg_samples)
# ## (2) Drug Discovery (DD1) samplar
# ### 항상 낮은 값 (e.g., IC50가 가능한 낮길 원함)
# class DD1_Samplar:
#     def __init__(self, seq_dataset, table_dataset):

#     # Sample a positive pair of input (except for input itself)
#     def pos_samplar(self, input_sample):

#         return self.sequence[rand_sample, :], rand_sample

#     # Sample a negative pair of input (except for input itself)
#     def neg_samplar(self, input_sample):

#         return self.sequence[rand_sample, :], rand_sample

#     def __call__(self, input_sample):

#         return (pos_samples, neg_samples)
# ## (3) Diet Design (DD2) samplar
# ### 항상 높은 값 (e.g., 영양점수가 가능한 높길 원함)
# class DD2_Samplar:
#     def __init__(self, seq_dataset, table_dataset):

#     # Sample a positive pair of input (except for input itself)
#     def pos_samplar(self, input_sample):

#         return self.sequence[rand_sample, :], rand_sample

#     # Sample a negative pair of input (except for input itself)
#     def neg_samplar(self, input_sample):

#         return self.sequence[rand_sample, :], rand_sample

#     def __call__(self, input_sample):

#         return (pos_samples, neg_samples)


# 토큰 샘플링 유형
def token_sampling(final_outputs, mode = 'greedy'):
    if mode == 'greedy':    
        pred_tokens = tf.expand_dims(tf.argmax(final_outputs[:, -1, :], -1), 1)

    elif mode == 'stochastic':
        prob_matrix = tf.nn.softmax(final_outputs[:, -1, :], -1)
        pred_tokens = tf.cast(np.apply_along_axis(lambda prob_vector: np.random.choice(a = final_outputs.shape[1] + 1, size = 1, p = prob_vector), arr = prob_matrix, axis = 1), dtype = tf.int64)

    return pred_tokens

