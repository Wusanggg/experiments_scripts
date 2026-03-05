import os
from pydub import AudioSegment
from tqdm import tqdm  # 用于显示进度条，如果没有请 pip install tqdm

# --- 路径配置 ---
# 背景音素材路径
BACKGROUND_PATH = r"E:\Dataset_mobicom\raw_data\background\residential_street.wav"

# 原始语音文件夹路径 (Speech Only)
SOURCE_DIR = r"E:\Dataset_mobicom\raw_data\speech\A\false"

# 叠加后的保存路径
OUTPUT_DIR = r"E:\Dataset_mobicom\sound_with_background\residential_street\speech\A\false"

# --- 增益配置 (SNR 控制) ---
# 背景音的增益调整（单位：dB）。
# 负值表示降低背景音音量。例如 -15 表示将背景音降低 15 分贝。
NOISE_GAIN_DB = 0

def batch_add_background():
    # 1. 检查并创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 2. 加载背景音素材
    if not os.path.exists(BACKGROUND_PATH):
        print(f"❌ 错误：找不到背景音文件 {BACKGROUND_PATH}")
        return

    print("正在加载背景音素材...")
    background = AudioSegment.from_wav(BACKGROUND_PATH)
    
    # 统一背景音参数 (16kHZ, 单声道) 以匹配之前的语音脚本
    background = background.set_frame_rate(16000).set_channels(1)
    
    # 调整背景音音量
    background = background + NOISE_GAIN_DB

    # 3. 遍历语音文件夹
    file_list = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".wav")]
    print(f"找到 {len(file_list)} 个语音文件，开始批量合成...")

    for file_name in tqdm(file_list, desc="处理进度"):
        speech_path = os.path.join(SOURCE_DIR, file_name)
        save_path = os.path.join(OUTPUT_DIR, file_name)

        try:
            # 加载语音文件
            speech = AudioSegment.from_wav(speech_path)
            speech = speech.set_frame_rate(16000).set_channels(1)

            # 叠加音频
            # 因为两者都是 10 秒，直接从 0ms 位置开始叠加即可
            combined = background.overlay(speech, position=0)

            # 导出最终文件
            combined.export(save_path, format="wav")
            
        except Exception as e:
            print(f"无法处理文件 {file_name}: {e}")

    print(f"\n✅ 处理完成！所有合成文件已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    batch_add_background()
