# Audio Agent Dataset & Evaluation

本仓库包含一套用于研究「耳机内置 Agent」场景的音频数据集构建与评估脚本，包括：

- **数据集构建**：TTS 合成语音、叠加真实背景噪声、拼接成长录音；
- **AST（音频事件分类模型）评估**：语音/非语音检测性能评估与阈值搜索；
- **Whisper 评估**：语音检测与转写性能评估；
- **Agent（大模型文本分类）评估**：在转写文本上进行 0/1 打标签，并与 ground truth 对比。

> 说明：示例路径中多为 Windows 路径，请根据自己的目录结构调整。

---

## 环境准备

建议使用 Python 3.9+，并在虚拟环境中安装依赖。

```bash
# 创建虚拟环境（Windows 示例）
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

部分脚本/Notebook 依赖 FFmpeg（`pydub`、`whisper` 会用到），请确保本机已安装并在 `PATH` 中。

---

## 仓库中主要文件

当前所有脚本和 Notebook 位于仓库根目录：

- Python 脚本：
  - `add_background.py`
  - `concat_wav.py`
  - `shuffle.py`
  - `comparing.py`
  - `label_with_openrouter.py`

- Notebook：
  - `voice_genereator.ipynb`
  - `accuracy.ipynb`
  - `detect_voice.ipynb`
  - `detect_voice_exam.ipynb`
  - `long_detection.ipynb`
  - `text_extract.ipynb`

- 示例数据：
  - `csv/` 目录下的若干 `*_results.csv`、`shuffled_*.csv` 等（你可以选择是否上传原始数据文件到 GitHub）。

---

## 各脚本与 Notebook 说明

### Python 脚本

- `add_background.py`  
  批量给干净语音 `.wav` 文件叠加指定背景噪声，统一为 **16 kHz 单声道**。  
  - 修改脚本顶部的：
    - `BACKGROUND_PATH`：背景噪声 WAV 路径
    - `SOURCE_DIR`：干净语音所在目录
    - `OUTPUT_DIR`：叠加背景后的输出目录
  - 然后运行：
    ```bash
    python add_background.py
    ```

- `concat_wav.py`  
  按指定逻辑拼接 WAV 文件，生成一个长音频：
  1. 先接入 `intro` 背景音；
  2. 再交错拼接 `nonspeech/` 下各子目录中的若干无语音片段；
  3. 最后按顺序拼接 `speech/<子目录>/true` 和 `speech/<子目录>/false` 中的语音。  
  支持命令行参数：
  ```bash
  python concat_wav.py \
    --intro <intro_wav路径> \
    --folder <主目录，内含nonspeech/speech> \
    --output-dir <输出目录> \
    --output-name <输出文件名(可选)> \
    --sr 16000
  ```

- `shuffle.py`  
  读取一个 CSV（当前示例指向 `csv/airport_results.csv`），**打乱所有行顺序** 后输出到新的 CSV（如 `csv/shuffled_airport.csv`），用于后续标注/评估时避免顺序偏差。

- `comparing.py`  
  对比两份带标签的 CSV：
  - 一份 ground truth（如 `shuffled_residential_street_gt.csv`）；
  - 一份 Agent 输出（如 `shuffled_residential_street_labeled_qwen.csv`）。  
  按 `Filename + Timestamp + Transcription` 精确对齐，输出：
  - 所有 `Label` 不一致的行；
  - 两个文件中 `Label` 的分布统计。  
  用于评估 Agent 文本打标签的性能。

- `label_with_openrouter.py`  
  使用 OpenRouter 上的 Qwen3.5-Flash 等模型，对 CSV 中某一列的转写文本逐行打 0/1 标签：
  - 1：与用户当前关注需求匹配（默认示例是“有人叫 Bob”）；
  - 0：与关注需求无关。  
  默认配置在脚本顶部（`INPUT_CSV`、`OUTPUT_CSV`、`REQUIREMENT` 等），也支持命令行参数修改。

  > **安全建议：**  
  > 不要在公共仓库中提交真实的 `OPENROUTER_API_KEY`。  
  > 推荐修改脚本，使其从环境变量中读取：
  > ```python
  > import os
  > OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
  > ```
  > 然后在本地运行前设置：
  > ```bash
  > # PowerShell 示例
  > $env:OPENROUTER_API_KEY="sk-or-xxxx"
  > python label_with_openrouter.py
  > ```

### Notebook

- `voice_genereator.ipynb`  
  使用 **Edge TTS** 合成多种英文语音：
  - 生成包含 “Bob” 的句子（正样本）与不包含 “Bob”（含干扰词，如 Rob、Pop 等）的句子（负样本）；
  - 随机选择不同英文发音人，为每条文本生成若干版本；
  - 将合成语音嵌入到固定长度（10 秒）的背景中；
  - 生成 `true/` 与 `false/` 两类数据集用于实验；
  - 还包含一个「声库审核」流程，为每个英文声库生成 10 秒样本，方便人工挑选合适的发音人。

- `accuracy.ipynb`  
  用 **AST 模型（MIT/ast-finetuned-audioset-10-10-0.4593）** 评估语音/非语音检测性能：
  - 将多条长录音按 10 秒切片；
  - 对每片段进行语音/非语音判断；
  - 与预先定义的真值序列对比，计算 Accuracy / Precision / Recall / F1、混淆矩阵；
  - 通过网格搜索调节概率阈值与最小 RMS 能量门限，寻找最优参数。

- `detect_voice.ipynb`  
  包含三个小实验：
  1. 使用 Whisper `small.en` 统计文件夹中哪些 wav 文件含有足够长的英语语音；
  2. 使用 AST 把某目录下的音频分为 Speech / Environment；
  3. 在一个场景（如室内公共大厅）上，用 AST 过滤出语音文件，再用 Distil-Whisper 做转写，并输出「文件名 + 类型 + 文本」。

- `detect_voice_exam.ipynb`  
  在合成的「Bob 检测」数据集上，评估：
  - AST 语音检测 + Distil-Whisper 转写 + 正则查找 `bob` 的整体性能；
  - 计算 TP / FP / TN / FN，给出 Accuracy / Precision / Recall 等；
  - 将每条样本的转写文本与预测结果写入日志，用于误差分析。

- `long_detection.ipynb`  
  针对单条长录音（如机场广播）：
  - 使用 AST 滑动窗口做语音检测；
  - 使用 Distil-Whisper 对检测出的语音片段转写；
  - 使用正则搜索包含航班号 `CA151` 的片段；
  - 输出各个命中的时间戳与对应文本。

- `text_extract.ipynb`  
  把长录音中的语音片段批量转写并导出为 CSV，用于后续 Agent 评估：
  - 先评估 AST 在长录音上的语音检测表现（与真值对比）；  
  - 然后对被判定为语音的 10 秒窗口执行转写，根据时间规则自动打 0/1 标签；
  - 最终生成若干 `*_results.csv`，字段包括 `Filename, Timestamp, Transcription, Label`。

---

## 实验流程

### 1. 制作数据集：TTS → 加背景噪声 → 拼成长串

1. **使用 `voice_genereator.ipynb` 生成 TTS 语音**
   - 打开 Notebook，按顺序运行单元；
   - Notebook 会调用 Edge TTS 生成多种发音人的英文语音，区分「含 Bob」与「不含 Bob」；
   - 输出的 wav 文件会写入 Notebook 中配置的目录（如 `A/true`, `A/false` 或 `test_experiment/true`, `test_experiment/false` 等）。

2. **使用 `add_background.py` 添加真实背景噪声**
   - 根据自己的数据修改脚本顶部的：
     - `BACKGROUND_PATH`：背景噪声 WAV（如街道、公交车内等）；
     - `SOURCE_DIR`：TTS 生成的干净语音目录（如 `A/true` / `A/false`）；
     - `OUTPUT_DIR`：加噪后的输出目录。
   - 在仓库根目录运行：
     ```bash
     python add_background.py
     ```

3. **使用 `concat_wav.py` 拼接为长录音**
   - 根据你的目录结构准备好：
     - `nonspeech/`：无语音片段子目录；
     - `speech/<场景>/true`：语音正样本；
     - `speech/<场景>/false`：语音负样本。
   - 修改命令行参数或脚本默认参数，使其指向正确路径；
   - 在仓库根目录运行，例如：
     ```bash
     python concat_wav.py \
       --intro E:\Dataset_mobicom\raw_data\background\residential_street.wav \
       --folder E:\Dataset_mobicom\sound_with_background\residential_street \
       --output-dir E:\Dataset_mobicom\evaluation
     ```
   - 得到用于后续评估的长录音（如 `evaluation/residential_street.wav`）。

---

### 2. 评估 AST 性能

1. 打开 `accuracy.ipynb`；
2. 确认 `BASE_PATH` 和 `files_config` 中的文件名与你实际的长录音路径一致（如 `evaluation/airport.wav` 等）；
3. 按顺序运行所有单元：
   - Notebook 会为每条长录音按 10 秒切片，用 AST 做语音/非语音检测；
   - 与预设真值对比，输出整体指标和各文件准确率；
   - 在网格搜索部分，可以得到推荐的 `SPEECH_THRESHOLD` 与 `MIN_RMS`。

---

### 3. 评估 Whisper 性能

根据需要选择以下 Notebook 之一或全部运行：

- **粗粒度检测 + 示例转写**：`detect_voice.ipynb`
- **在 Bob 实验集上的检测+转写评估**：`detect_voice_exam.ipynb`
- **在长录音中精确查找包含航班号等关键词的片段**：`long_detection.ipynb`

它们都会用到 Hugging Face 上的 `distil-whisper/distil-large-v3` 模型，请确保网络访问畅通。

---

### 4. 评估 Agent（大模型文本打标签）性能

完整流程如下：

1. **用 `text_extract.ipynb` 生成带标签的转写 CSV**
   - 打开 `text_extract.ipynb`；
   - 确认 `BASE_PATH` 和目标文件列表与你的长录音一致；
   - 运行 Notebook，生成若干 `*_results.csv`，包括字段：
     - `Filename, Timestamp, Transcription, Label`。

2. **使用 `shuffle.py` 打乱行顺序，构建测试集**
   - 修改 `shuffle.py` 中的输入/输出路径（默认是 `csv/airport_results.csv` → `csv/shuffled_airport.csv`）；
   - 运行：
     ```bash
     python shuffle.py
     ```
   - 手动复制打乱后的文件：
     - `shuffled_xxx.csv` → `shuffled_xxx_gt.csv`（保留 `Label`，作为 ground truth）；
     - `shuffled_xxx.csv` → `shuffled_xxx_unlabel.csv`，并**删除 `Label` 列**，作为 Agent 的输入。

3. **运行 `label_with_openrouter.py` 调用大模型打标签**
   - 修改脚本顶部或通过命令行传参，使：
     - `INPUT_CSV` 指向 `shuffled_xxx_unlabel.csv`；
     - `OUTPUT_CSV` 指向你希望输出的文件名（如 `shuffled_xxx_labeled_qwen.csv`）；
     - `REQUIREMENT` 描述当前关注需求，例如：
       > "Notify me if someone calls my name, Bob."
   - 配好 OpenRouter API Key（推荐用环境变量），在仓库根目录运行：
     ```bash
     python label_with_openrouter.py
     ```

4. **使用 `comparing.py` 对比 Agent 输出与 ground truth**
   - 修改 `comparing.py` 中的路径：
     - `df_gt` 指向 `shuffled_xxx_gt.csv`；
     - `df_labeled` 指向 `shuffled_xxx_labeled_*.csv`；
   - 运行：
     ```bash
     python comparing.py
     ```
   - 脚本会输出：
     - 所有 `Label` 不一致的行（含时间戳和转写文本）；
     - ground truth 与 Agent 输出的标签分布统计。

---
