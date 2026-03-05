# -*- coding: utf-8 -*-
"""
WAV 拼接脚本
逻辑：intro 开头 -> nonspeech 交错拼接 -> speech 各子文件夹内 true 再 false
"""

import os
import argparse
import numpy as np
from pathlib import Path

try:
    import scipy.io.wavfile as wavfile
    HAS_SCIPY = True
except ImportError:
    import wave
    import struct
    HAS_SCIPY = False


def read_wav_scipy(path):
    """用 scipy 读取 WAV，返回 (sample_rate, data), data 为 float32 [-1,1]"""
    rate, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return rate, data


def read_wav_wave(path):
    """用标准库 wave 读取 WAV"""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if sampwidth == 2:  # 16-bit
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    data = np.frombuffer(raw, dtype=dtype)
    data = data.astype(np.float32) / (2 ** (sampwidth * 8 - 1))
    data = data.reshape(-1, nch)
    return rate, data


def read_wav(path):
    if HAS_SCIPY:
        return read_wav_scipy(path)
    return read_wav_wave(path)


def write_wav_scipy(path, rate, data):
    """写入 WAV，data 为 float32 [-1,1]"""
    data = np.clip(data, -1.0, 1.0)
    data = (data * 32767).astype(np.int16)
    if data.shape[1] == 1:
        data = data.squeeze(axis=1)
    wavfile.write(path, rate, data)


def write_wav_wave(path, rate, data):
    data = np.clip(data, -1.0, 1.0)
    data = (data * 32767).astype(np.int16)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    nch = data.shape[1]
    with wave.open(path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def write_wav(path, rate, data):
    if HAS_SCIPY:
        return write_wav_scipy(path, rate, data)
    return write_wav_wave(path, rate, data)


def resample_to_rate(data, orig_rate, target_rate):
    """简单线性插值重采样到 target_rate"""
    if orig_rate == target_rate:
        return data
    duration = data.shape[0] / orig_rate
    new_len = int(duration * target_rate)
    x_old = np.linspace(0, 1, data.shape[0], endpoint=False)
    x_new = np.linspace(0, 1, new_len, endpoint=False)
    out = np.zeros((new_len, data.shape[1]), dtype=data.dtype)
    for c in range(data.shape[1]):
        out[:, c] = np.interp(x_new, x_old, data[:, c])
    return out


def match_channels(data, target_channels):
    """将 data 变为 target_channels 声道"""
    if data.shape[1] == target_channels:
        return data
    if data.shape[1] == 1 and target_channels == 2:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] == 2 and target_channels == 1:
        return (data[:, 0:1] + data[:, 1:2]) / 2.0
    return data


def collect_and_concat(
    intro_wav_path,
    folder_path,
    output_dir,
    output_basename=None,
    target_sr=16000,
):
    """
    intro_wav_path: 开头 WAV 路径
    folder_path: 主目录（其下可能有 nonspeech、speech）
    output_dir: 输出目录
    output_basename: 输出文件名（不含扩展名），默认用 folder_path 最后一级目录名
    target_sr: 统一采样率，None 则使用 intro 的采样率
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_basename is None:
        output_basename = folder_path.name
    out_path = output_dir / f"{output_basename}.wav"

    # 1. 读取 intro
    if not os.path.isfile(intro_wav_path):
        raise FileNotFoundError(f"Intro WAV 不存在: {intro_wav_path}")
    rate_intro, data_intro = read_wav(intro_wav_path)
    if target_sr is None:
        target_sr = rate_intro
    if rate_intro != target_sr:
        data_intro = resample_to_rate(data_intro, rate_intro, target_sr)
    nch = data_intro.shape[1]
    chunks = [data_intro]

    # 2. nonspeech：子文件夹各 50 个 wav，按 1_1,2_1,...,N_1, 1_2,2_2,...,N_2, ... 1_50,...,N_50
    nonspeech_dir = folder_path / "nonspeech"
    if nonspeech_dir.is_dir():
        subdirs = sorted([d for d in nonspeech_dir.iterdir() if d.is_dir()])
        if subdirs:
            # 每个子文件夹内 wav 按文件名排序，取 50 个
            file_lists = []
            for d in subdirs:
                files = sorted(d.glob("*.wav"))[:50]
                if len(files) < 50:
                    print(f"警告: {d} 仅 {len(files)} 个 wav，不足 50")
                file_lists.append(files)
            for i in range(50):
                for flist in file_lists:
                    if i < len(flist):
                        r, d = read_wav(str(flist[i]))
                        if r != target_sr:
                            d = resample_to_rate(d, r, target_sr)
                        d = match_channels(d, nch)
                        chunks.append(d)

    # 3. speech：每个子文件夹内有 true、false，先全部 true 再全部 false
    speech_dir = folder_path / "speech"
    if speech_dir.is_dir():
        subdirs = sorted([d for d in speech_dir.iterdir() if d.is_dir()])
        for sub in subdirs:
            true_dir = sub / "true"
            false_dir = sub / "false"
            for subsub_name, subsub in [("true", true_dir), ("false", false_dir)]:
                if not subsub.is_dir():
                    continue
                files = sorted(subsub.glob("*.wav"))
                for f in files:
                    r, d = read_wav(str(f))
                    if r != target_sr:
                        d = resample_to_rate(d, r, target_sr)
                    d = match_channels(d, nch)
                    chunks.append(d)

    # 4. 拼接并写入
    if len(chunks) == 0:
        raise ValueError("没有可拼接的音频段")
    concat = np.concatenate(chunks, axis=0)
    write_wav(str(out_path), target_sr, concat)
    print(f"已保存: {out_path} (采样率 {target_sr} Hz, 总样本数 {concat.shape[0]})")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="按指定逻辑拼接 WAV：intro + nonspeech 交错 + speech")
    parser.add_argument(
        "--intro",
        type=str,
        default=r"E:\Dataset_mobicom\raw_data\background\residential_street.wav",
        help="开头 WAV 路径",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=r"E:\Dataset_mobicom\sound_with_background\residential_street",
        help="主目录（内含 nonspeech / speech）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"E:\Dataset_mobicom\evaluation",
        help="输出目录",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="输出文件名（不含 .wav），默认用 folder 最后一级目录名",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="统一采样率（默认 16000），设为 0 表示使用 intro 的采样率",
    )
    args = parser.parse_args()
    target_sr = None if args.sr == 0 else args.sr
    collect_and_concat(
        args.intro,
        args.folder,
        args.output_dir,
        output_basename=args.output_name,
        target_sr=target_sr,
    )


if __name__ == "__main__":
    main()
