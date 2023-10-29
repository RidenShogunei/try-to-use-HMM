from HMM import HMM
from voiceprocess import voiceprocess
import pickle

# 处理两段音频并提取MFCC特征
mfcc_1 = voiceprocess("voice/1.wav")
mfcc_1 = mfcc_1.flatten()
mfcc_2 = voiceprocess("voice/2.wav")
mfcc_2 = mfcc_2.flatten()

# 将MFCC特征序列作为观测序列准备好
observations = [mfcc_1, mfcc_2]

# 对模型的声明
n_state = 100  # 你只有两种语音，因此隐状态的个数为2
n_observation = 2  # 观测状态的个数，等于一个MFCC特征向量的维度

# 创建并训练HMM模型
hmm_model = HMM(n_state, n_observation)
hmm_model.train(observations, iterations=1000)

# 保存模型到文件
with open('hmm_model.pkl', 'wb') as model_file:
    pickle.dump(hmm_model, model_file)

print("模型已保存到文件'hmm_model.pkl'中")


