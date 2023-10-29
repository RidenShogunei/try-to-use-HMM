import librosa
import numpy as np

def voiceprocess(path):
    # 加载音频文件
    audio_path = path
    audio, sr = librosa.load(audio_path, sr=None)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # 归一化到 0 到 255 的范围内
    normalized_mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc)) * 255

    # 将浮点数转换为整数
    observation = normalized_mfcc

    # 打印MFCC特征的形状
    #print("MFCC shape:", observation.shape)

    # 输出MFCC特征矩阵
    #print("MFCC matrix:\n", observation)

    return observation
