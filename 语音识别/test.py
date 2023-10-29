from HMM import HMM
from voiceprocess import voiceprocess
import pickle

# 加载模型
with open('hmm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# 使用加载的模型进行预测
new_mfcc = voiceprocess("voice/new.wav")  # 处理新的音频并提取MFCC特征
new_mfcc = new_mfcc.flatten()
print(new_mfcc.shape)
predicted_states = loaded_model.predict(new_mfcc)

print("预测的隐状态序列:", predicted_states)