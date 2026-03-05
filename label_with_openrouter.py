import os
import csv
import time
import argparse
import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# 在本地安全环境下使用时，可以在这里直接填写你的 OpenRouter API Key。
# 强烈建议：不要把填写好真实 Key 的脚本提交到任何远程仓库或分享给他人。
OPENROUTER_API_KEY = "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"

# 当前任务的默认配置：输入 CSV、输出目录、文本列（第三列=索引2）、监测需求
INPUT_CSV = r"D:\Desktop\audio\zhenyu\original_dataset\sound_with_background\csv\shuffled_residential_street_unlabel.csv"
OUTPUT_DIR = r"D:\Desktop\audio\zhenyu\original_dataset\sound_with_background"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "shuffled_residential_street_labeled_qwen.csv")
TEXT_COL_INDEX = 2   # 第三列（0-based）
REQUIREMENT = "Notify me if someone calls my name, Bob."
HAS_HEADER = True    # 若 CSV 无表头可改为 False


def call_openrouter(api_key: str, text: str, requirement: str, model: str = "qwen/qwen3.5-flash-02-23", retries: int = 3) -> str:
    """
    调用 OpenRouter 的模型（默认 qwen/qwen3.5-flash-02-23，对应 Qwen3.5-Flash），对单条文本进行 0/1 分类。
    返回字符串 "0" 或 "1"（默认失败时返回 "0"）。
    """
    system_prompt = (
        "You are an Agent built into the user's earphones. "
        "The earphones capture ambient sounds, extract speech, and transcribe it into text, which is then sent to you. "
        "You know the user's current focus requirement. For each piece of text, your job is to decide whether it "
        "matches the user's requested focus. If it matches, output 1. If it does not match, output 0.\n\n"
        "Rules:\n"
        "1. Consider only whether the text is relevant/important to the user's focus.\n"
        "2. If the text is clearly unrelated or only loosely related, output 0.\n"
        "3. Respond with ONLY a single character: 1 or 0. No explanation, no extra text."
    )

    user_prompt = (
        f"User's current focus requirement (what the user wants to monitor in this scenario):\n"
        f"{requirement}\n\n"
        f"Transcript to evaluate:\n"
        f"{text}\n\n"
        f"Does this transcript match the user's requested focus? "
        f"Answer strictly with 1 (match) or 0 (no match)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # 下面两个可选，但官方推荐提供
        "HTTP-Referer": "https://example.com",  # 换成你自己的站点或留空字符串
        "X-Title": "csv-labeling-earphone-agent",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    # 连接与读取分开设超时，避免长时间无响应像“卡住”
    connect_timeout = 15
    read_timeout = 45

    for attempt in range(retries):
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=(connect_timeout, read_timeout),
            )
            if not resp.ok:
                err_detail = resp.text
                try:
                    err_detail = resp.json()
                except Exception:
                    pass
                raise RuntimeError(f"OpenRouter API 错误 {resp.status_code}: {err_detail}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # 只取第一个出现的 '0' 或 '1'
            for ch in content:
                if ch in ("0", "1"):
                    return ch

            # 如果模型没有按要求返回，保守输出 0
            return "0"
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  请求异常（{e}），{wait}s 后重试 ({attempt + 1}/{retries})...", flush=True)
                time.sleep(wait)
            else:
                print(f"调用 OpenRouter 失败（已重试 {retries} 次），错误：{e}", flush=True)
                return "0"


def label_csv(
    input_path: str,
    output_path: str,
    api_key: str,
    requirement: str,
    text_col_index: int = 0,
    has_header: bool = False,
    model: str = "qwen/qwen3.5-flash-02-23",
):
    """
    读取 input_path 的 CSV，对每行指定列做 0/1 打标，并把结果写入 output_path。
    - text_col_index: 文本所在列的索引（从 0 开始），默认第 1 列。
    - has_header: 是否有表头行，若有则在最后增加新列名 'label'。
    """
    with open(input_path, "r", encoding="utf-8-sig", newline="") as fin, \
            open(output_path, "w", encoding="utf-8-sig", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        first_row = True
        row_index = 0  # 当前数据行序号（不含表头），用于进度显示
        for row in reader:
            if first_row and has_header:
                # 表头增加新列
                row.append("Label")
                writer.writerow(row)
                first_row = False
                continue

            first_row = False
            row_index += 1

            if len(row) <= text_col_index:
                # 这一行列不够，保守打 0
                row.append("0")
                writer.writerow(row)
                print(f"  [{row_index}] 跳过（列不足），标为 0", flush=True)
                continue

            text = row[text_col_index]
            print(f"  [{row_index}] 正在请求 API...", end=" ", flush=True)
            label = call_openrouter(api_key, text, requirement, model=model)
            row.append(label)
            writer.writerow(row)
            print(f"label={label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="使用 OpenRouter 对 CSV 语音转写文本打 0/1 标签。")
    parser.add_argument("--input", "-i", default=INPUT_CSV, help="输入 CSV 文件路径")
    parser.add_argument("--output", "-o", default=OUTPUT_CSV, help="输出 CSV 文件路径")
    parser.add_argument("--requirement", "-r", default=REQUIREMENT, help="当前场景下你想要监测/关注的信息描述")
    parser.add_argument("--text-col-index", "-c", type=int, default=TEXT_COL_INDEX, help="转写文本所在的列索引（从 0 开始）")
    parser.add_argument("--has-header", action="store_true", default=HAS_HEADER, help="CSV 第一行是否为表头")
    parser.add_argument("--no-header", action="store_true", help="CSV 无表头（与 --has-header 二选一）")
    parser.add_argument("--model", default="qwen/qwen3.5-flash-02-23", help="OpenRouter 模型 ID，如 qwen/qwen3.5-flash-02-23")

    args = parser.parse_args()
    has_header = args.has_header and not args.no_header

    api_key = OPENROUTER_API_KEY
    if not api_key or api_key == "YOUR_OPENROUTER_API_KEY_HERE":
        raise RuntimeError(
            "请先在脚本顶部的 OPENROUTER_API_KEY 常量中填入你的真实 OpenRouter API Key。"
            "注意：仅在本地、安全环境使用，且不要把包含真实 Key 的脚本提交到远程仓库或分享给他人。"
        )

    print(f"输入: {args.input}", flush=True)
    print(f"输出: {args.output}", flush=True)
    print(f"模型: {args.model}，监测需求: {args.requirement}", flush=True)
    print("开始标注（每行会请求一次 API，请耐心等待）...", flush=True)

    label_csv(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        requirement=args.requirement,
        text_col_index=args.text_col_index,
        has_header=has_header,
        model=args.model,
    )

    print("标注完成！输出文件：", args.output)


if __name__ == "__main__":
    main()