import os
import pyaudio
import wave
import datetime
# 配置录音参数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 声道数
RATE = 44100  # 采样率
CHUNK = 1024  # 每次读取的音频帧大小
RECORD_SECONDS = 2  # 录音时长
OUTPUT_FOLDER = 'voice'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 初始化PyAudio
audio = pyaudio.PyAudio()

# 打开音频流
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("开始录音...")

# 录制音频数据
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束.")

# 停止音频流
stream.stop_stream()
stream.close()
audio.terminate()
# 使用datetime生成带有时间戳的文件名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化时间戳
output_filename = os.path.join(OUTPUT_FOLDER, f"recording_{current_time}.wav")
# 生成新的音频文件名

# 保存录音数据到文件
wf = wave.open(output_filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("音频保存成功:", output_filename)