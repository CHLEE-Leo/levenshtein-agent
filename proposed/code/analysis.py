# %%
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as  mpatches
import pickle
import json
from sklearn.manifold import TSNE
from utils import *
from model import Reward_Function, AETransformer

target_task = 'ST'
num_epochs = str(1000)
batch_size = str(1024)

# 실험 데이터 로드
test_input_sequence = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/input_sequence(test).npy')
eos_idx = indexing_eos_token(test_input_sequence)
test_input_sequence = test_input_sequence[np.where(eos_idx >= 4)[0], :]                 # 문장의 [eos] 토큰의 인덱스가 4 이상인 시퀀스만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)
test_attribute = np.load('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer/attribute(test).npy')
test_attribute = test_attribute[np.where(eos_idx >= 4)[0]]                              # 문장의 [eos] 토큰의 인덱스가 4 이상인 경우만 필터링 (= [bos] & [eos] 제외 토큰 갯수가 3개 미만인 시퀀스 필터링)

# 토큰 사전 가져오기
with open('/home/messy92/Leo/NAS_folder/ICML23/prep_data/text-style-transfer' + '/token_dict.pickle', 'rb') as f:
    token_dict = pickle.load(f)
special_token_list = ['[pad]', '[mask]']
reward_class_token_list = ['[' + 'R_' + str(reward_class) + ']' for reward_class in range(len(np.unique(test_attribute)))]
edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
add_token_list = special_token_list + reward_class_token_list + edit_token_list
token_dict = add_token_list_in_dict(add_token_list, token_dict)
action_set = list(token_dict.values())[-len(edit_token_list):]

# "리버스" 프롬프트 벡터 (= 보상 클래스 토큰 + [BOS] 토큰) 생성 준비
test_reward_class_vector = np.ones(shape = test_attribute.shape).astype(np.int32)
for reward_class in range(len(np.unique(test_attribute))):
    rev_reward_class = np.unique(test_attribute)[np.where(reward_class != np.unique(test_attribute))[0]][0]
    test_reward_class_vector[np.where(test_attribute == reward_class)[0]] = get_token(token_dict, '[R_' + str(rev_reward_class) + ']')

# 프롬프트 벡터 생성
test_prompt_vector = test_reward_class_vector[:, np.newaxis]


# 마스킹 된 테스트 셋만들기
mean_num_mask = get_masking_param(test_input_sequence)    # 인풋 시퀀스 마스킹 파라미터인 mean_num_mask (= 평균 마스크 갯수) 정의
mask_lev = mask_level_seletor(thredshold = 0.5)             # token-level masking vs. span-level masking
masked_test_input_sequence, masking_idx = poisson_mask_generator(test_input_sequence, lambda_ = mean_num_mask, token_dict = token_dict, masking_level = mask_lev)


# 1) 보상함수 모델 초기화
# --> 속성 보상함수 관련 파라미터 로드
with open('/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/Reward_Function/kwargs_attr_reward_function_mlp_500', 'r') as f:
    re_kwargs = json.load(f)
# with open('/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/Reward_Function/kwargs_attr_reward_function_ST_1000_1024', 'r') as f:
#     re_kwargs = json.load(f)
attr_reward_function = Reward_Function(len(np.unique(test_attribute)), **re_kwargs)

# --> 사전학습된 속성 보상함수 로드
load_dir1 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/attr_reward_function_mlp_500'
# load_dir1 = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/attr_reward_function_ST_1000_1024'
print(load_dir1)
attr_reward_function.load_weights(tf.train.latest_checkpoint(load_dir1))

# ---------------------------------------------------------------------------------------- #
# 2) Env_Gen 모델 초기화
env_gen_kwargs_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/NAR/kwargs_NAR_ST_500'
# env_gen_kwargs_dir = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/NAR/kwargs_coBART_ST_500'
# env_gen_model_name = 'NAR'
# prefix = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/' + env_gen_model_name
# env_gen_kwargs = '/kwargs_' + env_gen_model_name + '_' + str(target_task) + '_500'
# env_gen_kwargs_dir = prefix + env_gen_kwargs
# env_gen_kwargs = '/kwargs_' + env_gen_model_name + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
# env_gen_kwargs_dir = prefix + env_gen_kwargs
with open(env_gen_kwargs_dir, 'r') as f:
    env_kwargs = json.load(f)

env_gen_model = AETransformer(**env_kwargs)


# --> 초기 정책모델 학습 가중치 로드하여 타겟 정책모델 가중치 초기화
env_gen_weights_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/NAR_ST_500'
# env_gen_weights_dir = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/coBART_ST_500'
# prefix = '/home/messy92/Leo/NAS_folder/ICML23/weights/text-style-transfer/'
# env_gen_weights_dir = prefix + 'coBART_ST' + '_500'
# env_gen_weights_dir = prefix + env_gen_model_name + '_' + str(target_task) + '_500'
# env_gen_weights_dir = prefix + env_gen_model_name + '_' + str(target_task) + '_' + str(num_epochs) + '_' + str(batch_size)
print(env_gen_weights_dir)
env_gen_model.load_weights(tf.train.latest_checkpoint(env_gen_weights_dir))

'''
# ------------------------------------------------ 원 버전 ----------------------------------------------------------------------------
'''
'''
원버전 - 잠재벡터 봅기
'''
# prompt_inputs = np.concatenate([test_prompt_vector, test_input_sequence], axis = -1)
# _ ,_, dec_outputs, _ = env_gen_model((test_input_sequence, prompt_inputs), training = False)
_ ,_, dec_outputs, _ = env_gen_model((test_input_sequence, test_input_sequence), training = False)
gen_seqs = tf.cast(tf.math.argmax(dec_outputs, axis = -1), dtype = tf.int32)
enc_pad_mask, _, _ = attr_reward_function.mask_generator(gen_seqs, gen_seqs)
attr_rewards = tf.nn.softmax(attr_reward_function(gen_seqs, enc_pad_mask, training = False), axis = -1)

env_gen_model.encoder.embedding_layer.weights
mle_z = env_gen_model.encoder.embedding_layer(test_input_sequence)
mle_z2 = tf.reshape(mle_z, shape = (mle_z.shape[0], -1))
attr_reward_function.encoder.embedding_layer.weights
re_z = attr_reward_function.encoder.embedding_layer(test_input_sequence)
re_z2 = tf.reshape(re_z, shape = (re_z.shape[0], -1))

'''
원버전 - T-SNE 값 뽑기
'''
tsne_random_seed = 700
tsne_num_components = 2
column_list = ['pc_' + str(i) for i in range(tsne_num_components)]
mle_tsne = TSNE(n_components = tsne_num_components, random_state=tsne_random_seed, perplexity = 20)
mle_tsne_result = mle_tsne.fit_transform(mle_z2)
mle_tsne_result_df = pd.DataFrame(data = mle_tsne_result, columns = column_list)
mle_tsne_result_df['Objective'] = np.zeros(shape = (mle_tsne_result_df.shape[0])).astype('int')

re_tsne = TSNE(n_components = tsne_num_components, random_state=tsne_random_seed, perplexity = 20)
re_tsne_result = re_tsne.fit_transform(re_z2)
re_tsne_result_df = pd.DataFrame(data = re_tsne_result, columns = column_list)
re_tsne_result_df['Objective'] = np.ones(shape = (re_tsne_result_df.shape[0])).astype('int')

'''
마스킹 버전 - 잠재벡터 봅기
'''
# prompt_inputs = np.concatenate([test_prompt_vector, masked_test_input_sequence], axis = -1)  #   마스크 인풋 시퀀스의 앞에 프롬프트 붙여주기
# _ ,_, dec_outputs, _ = env_gen_model((masked_test_input_sequence, prompt_inputs), training = False)
_ ,_, dec_outputs, _ = env_gen_model((masked_test_input_sequence, masked_test_input_sequence), training = False)
env_gen_model.encoder.embedding_layer.weights
mask_mle_z = env_gen_model.encoder.embedding_layer(masked_test_input_sequence)
mask_mle_z2 = tf.reshape(mask_mle_z, shape = (mask_mle_z.shape[0], -1))

'''
마스킹 버전 - T-SNE 값 뽑기
'''
tsne_num_components = 2
column_list = ['pc_' + str(i) for i in range(tsne_num_components)]
mask_mle_tsne = TSNE(n_components = tsne_num_components, random_state=tsne_random_seed, perplexity = 20)
mask_mle_tsne_result = mask_mle_tsne.fit_transform(mask_mle_z2)
mask_mle_tsne_result_df = pd.DataFrame(data = mask_mle_tsne_result, columns = column_list)
mask_mle_tsne_result_df['Objective'] = np.zeros(shape = (mask_mle_tsne_result_df.shape[0])).astype('int')

# 그림 뽑기
target_figure = input('input : likelihood, reward, pseudo-likelihood or overlap')
if target_figure == 'likelihood' or target_figure == 'reward':

    '''
    원버전 - kdeplot 그리기
    '''
    all_tsne_result_df = pd.concat([mle_tsne_result_df, re_tsne_result_df], axis = 0)
    mpl.rc('font', family='DejaVu Sans') #한글 폰트 설정
    label_list = ["Likelihood", "Reward-pos", "Reward-neg"]
    all_tsne_result_df_replace = all_tsne_result_df.replace({'Objective' : {i: label for i, label in enumerate(label_list)} })

    # levels_vector = [0.55, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0]
    levels_vector = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_level = 0.5

    if target_figure == 'likelihood':

        # likelihood만 그릴 떄
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#ECFFFF")
        p = sns.kdeplot(x = all_tsne_result_df.iloc[:1000, 0], y = all_tsne_result_df.iloc[:1000, 1], color = 'black', n_levels=5, linestyles=":", alpha=0.5)
        sns.kdeplot(x = all_tsne_result_df.iloc[:1000, 0], y = all_tsne_result_df.iloc[:1000, 1], color = '#2ca02c', fill = True, label = 'Likelihood',
                            levels=levels_vector, alpha = alpha_level)

        p.axes.set_xlabel('Latent direction 0', fontsize = 18)
        p.axes.set_xlim(-52, 55)
        p.axes.set_ylabel('Latent direction 1', fontsize = 18)
        p.axes.set_ylim(-60, 60)
        plt.grid(False)

        # likelihood만 그릴 때
        handles = [mpatches.Patch(facecolor='#2ca02c', label="Likelihood")]

        plt.legend(title = 'Objective', handles=handles, prop = {'size':13})
        # plt.show()
        plt.savefig('/home/messy92/Leo/NAS_folder/ICML23/' + target_figure + '.png', dpi = 330)

    elif target_figure == 'reward':

        # reward만 그릴 때
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#ECFFFF")
        p = sns.kdeplot(x = all_tsne_result_df.iloc[1000:, 0], y = all_tsne_result_df.iloc[1000:, 1], color = 'black', n_levels=5, linestyles=":", alpha=0.5)
        sns.kdeplot(x = all_tsne_result_df.iloc[1000:1500, 0], y = all_tsne_result_df.iloc[1000:1500, 1], color = '#d62728', fill = True, label = 'Reward-neg',
                            levels=levels_vector, alpha = alpha_level)
        sns.kdeplot(x = all_tsne_result_df.iloc[1500:2000, 0], y = all_tsne_result_df.iloc[1500:2000, 1], color = '#1f77b4', fill = True, label = 'Reward-pos',
                            levels=levels_vector, alpha = alpha_level)

        p.axes.set_xlabel('Latent direction 0', fontsize = 18)
        p.axes.set_xlim(-52, 55)
        p.axes.set_ylabel('Latent direction 1', fontsize = 18)
        p.axes.set_ylim(-60, 60)
        plt.grid(False)

        # reward만 그릴 때
        handles = [mpatches.Patch(facecolor='#d62728', label="Reward-neg"),
                mpatches.Patch(facecolor='#1f77b4', label="Reward-pos")]

        plt.legend(title = 'Objective', handles=handles, prop = {'size':13})
        # plt.show()
        plt.savefig('/home/messy92/Leo/NAS_folder/ICML23/' + target_figure + '.png', dpi = 330)

elif target_figure == 'pseudo-likelihood':

    '''
    마스킹 버전 - kdeplot 그리기
    '''
    all_tsne_result_df = pd.concat([mask_mle_tsne_result_df, re_tsne_result_df], axis = 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ECFFFF")
    mpl.rc('font', family='DejaVu Sans') #한글 폰트 설정
    label_list = ["Pseudo-Likelihood", "Reward-neg", "Reward-pos"]
    all_tsne_result_df_replace = all_tsne_result_df.replace({'Objective' : {i: label for i, label in enumerate(label_list)} })

    levels_vector = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_level = 0.5

    # pseudo-likelihood만 그릴 때
    p = sns.kdeplot(x = all_tsne_result_df.iloc[:1000, 0], y = all_tsne_result_df.iloc[:1000, 1], color = 'black', n_levels=5, linestyles=":", alpha=0.5)
    sns.kdeplot(x = all_tsne_result_df.iloc[:1000, 0], y = all_tsne_result_df.iloc[:1000, 1], color = '#8172b3', fill = True, label = 'Pseudo-Likelihood',
                        levels=levels_vector, alpha = alpha_level)

    # # reward만 그릴 때
    # p = sns.kdeplot(x = all_tsne_result_df.iloc[1000:, 0], y = all_tsne_result_df.iloc[1000:, 1], color = 'black', n_levels=5, linestyles=":", alpha=0.5)
    # sns.kdeplot(x = all_tsne_result_df.iloc[1000:1500, 0], y = all_tsne_result_df.iloc[1000:1500, 1], color = '#d62728', fill = True, label = 'Reward-neg',
    #                        levels=levels_vector, alpha = alpha_level)
    # sns.kdeplot(x = all_tsne_result_df.iloc[1500:2000, 0], y = all_tsne_result_df.iloc[1500:2000, 1], color = '#1f77b4', fill = True, label = 'Reward-pos',
    #                        levels=levels_vector, alpha = alpha_level)

    p.axes.set_xlabel('Latent direction 0', fontsize = 18)
    p.axes.set_xlim(-52, 55)
    p.axes.set_ylabel('Latent direction 1', fontsize = 18)
    p.axes.set_ylim(-60, 60)
    plt.grid(False)

    # # 모두 그릴 때
    # handles = [mpatches.Patch(facecolor='#2ca02c', label="Likelihood"),
    #            mpatches.Patch(facecolor='#d62728', label="Reward-neg"),
    #            mpatches.Patch(facecolor='#1f77b4', label="Reward-pos")]

    # likelihood만 그릴 때
    handles = [mpatches.Patch(facecolor='#8172b3', label="Pseudo-Likelihood")]

    # # reward만 그릴 때
    # handles = [mpatches.Patch(facecolor='#d62728', label="Reward-neg"),
    #            mpatches.Patch(facecolor='#1f77b4', label="Reward-pos")]
    plt.legend(title = 'Objective', handles=handles, prop = {'size':13})
    # plt.show()
    plt.savefig('/home/messy92/Leo/NAS_folder/ICML23/' + target_figure + '.png', dpi = 330)

elif target_figure == 'overlap':
    '''
    바이올린 또는 막대플롯으로 잠재공간에서의 mismatch 보여주기
    '''
    mle_tsne_result_df['Objective'] = np.zeros(shape = (mle_tsne_result_df.shape[0])).astype('int')
    mask_mle_tsne_result_df['Objective'] = np.ones(shape = (mask_mle_tsne_result_df.shape[0])).astype('int')

    re_tsne_result_df['Objective'] = np.ones(shape = (re_tsne_result_df.shape[0])).astype('int')
    re_tsne_result_df['Objective'][:500] = np.ones(shape = (re_tsne_result_df.shape[0]//2)).astype('int') * 2
    re_tsne_result_df['Objective'][500:] = np.ones(shape = (re_tsne_result_df.shape[0]//2)).astype('int') * 3

    all_tsne_result_df = pd.concat([mle_tsne_result_df, mask_mle_tsne_result_df], axis = 0)
    all_tsne_result_df = pd.concat([all_tsne_result_df, re_tsne_result_df], axis = 0)

    label_list = ["Likelihood", "Pseudo-Likelihood", "Reward-neg", "Reward-pos"]
    all_tsne_result_df_replace = all_tsne_result_df.replace({'Objective' : {i: label for i, label in enumerate(label_list)} })

    all_tsne_result_df_category = copy.deepcopy(all_tsne_result_df_replace)
    interval_num = 4
    all_tsne_result_df_category['pc_0'] = pd.cut(all_tsne_result_df['pc_0'], interval_num)
    interval_list = list(all_tsne_result_df_category.pc_0.cat.categories.values.to_tuples())
    # all_tsne_result_df_category['pc_1'] = pd.cut(all_tsne_result_df['pc_1'], interval_num)
    # interval_list = list(all_tsne_result_df_category.pc_1.cat.categories.values.to_tuples())
    my_xticks = [str(i).replace(')', ']').replace(',', ',\n') for i in interval_list]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = sns.boxplot(data=all_tsne_result_df_category, x="pc_0", y="pc_1", hue = 'Objective', palette = ['#2ca02c', '#8172b3', '#d62728', '#1f77b4'])
    plt.legend(title = 'Objective', prop = {'size':13})
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    ax.set_xticklabels(labels=my_xticks)
    p.axes.set_xlabel('Latent direction 0', fontsize = 18)
    p.axes.set_ylabel('Latent direction 1', fontsize = 18)
    plt.xticks(ha='center')
    # ax.set_yticklabels(labels=my_xticks)
    # plt.yticks(ha='center')
    plt.tight_layout()
    plt.grid(False)
    # plt.show()
    plt.savefig('/home/messy92/Leo/NAS_folder/ICML23/' + target_figure + '.png', dpi = 330)    

# # 모두 그릴 때
# handles = [mpatches.Patch(facecolor='#2ca02c', label="Likelihood"),
#            mpatches.Patch(facecolor='#d62728', label="Reward-neg"),
#            mpatches.Patch(facecolor='#1f77b4', label="Reward-pos")]

'''
# ------------------------------------------------ 마스킹 버전 ----------------------------------------------------------------------------
'''
# del env_gen_model
# # 2) Env_Gen 모델 초기화
# env_gen_model_name = input('decoder (BART vs. NAR) : ')
# prefix = '/home/messy92/Leo/NAS_folder/ICML23/proposed/hyper-parameters/text-style-transfer/' + env_gen_model_name
# env_gen_kwargs = '/kwargs_' + env_gen_model_name + '_' + str(target_task) + '_500'
# env_gen_kwargs_dir = prefix + env_gen_kwargs
# with open(env_gen_kwargs_dir, 'r') as f:
#     env_kwargs = json.load(f)
# env_gen_model = AETransformer(**env_kwargs)






# %%
# columns = ["ACC", "BLEU_1", ""]
h = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/hamonic.csv')
g = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/geometric.csv')
a = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/arithemetic.csv')
s = pd.read_csv('/home/messy92/Leo/NAS_folder/ICML23/summation.csv')
all = pd.concat([h, g, a, s], axis = 0)

# all.eta = all.eta.astype('category')
all.reward = all.reward.astype('category')

bbb = all[(all["eta"] != 1.000)]
ccc = bbb[bbb["eta"] != 0.000]

sns.catplot(
    data = ccc,
    x = 'eta', y = 'ACC', hue = 'reward',
    linestyles=["-", "-", "-", "-"], kind="point"
)

sns.catplot(
    data = ccc,
    x = 'eta', y = 'BLEU_1', hue = 'reward',
    linestyles=["-", "-", "-", "-"], kind="point"
)

sns.catplot(
    data = ccc,
    x = 'eta', y = 'PPL (gpt_lr= 0.001)', hue = 'reward',
    linestyles=["-", "-", "-", "-"], kind="point"
)

sns.relplot(
    data=ccc, 
    x="BLEU_1", y="ACC", hue="reward"
)
