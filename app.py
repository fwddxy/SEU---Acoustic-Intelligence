"""中文 Streamlit 环境声音识别页面。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import DISPLAY_LABELS, MODEL_DIR, MODEL_NAMES, predict_features
from src.yamnet import analyze_audio


MODEL_LABELS = {
    "mfcc_svm": "MFCC + 支持向量机",
    "yamnet_svm": "YAMNet + 支持向量机",
    "yamnet_mlp": "YAMNet + 多层感知机",
}
MODEL_ACCURACY = {
    "mfcc_svm": "46.50%",
    "yamnet_svm": "83.00%",
    "yamnet_mlp": "81.25%",
}

CLASS_LABELS = {
    "dog": "狗叫声",
    "rooster": "公鸡打鸣",
    "rain": "雨声",
    "sea_waves": "海浪声",
    "crackling_fire": "炉火噼啪声",
    "crying_baby": "婴儿哭声",
    "sneezing": "喷嚏声",
    "clock_tick": "时钟滴答声",
    "helicopter": "直升机声",
    "chainsaw": "电锯声",
}

YAMNET_LABELS = {
    "Chainsaw": "电锯",
    "Helicopter": "直升机",
    "Rain": "雨声",
    "Sea waves": "海浪",
    "Crackling fire": "炉火噼啪声",
    "Crying, baby": "婴儿哭声",
    "Sneeze": "喷嚏",
    "Tick": "滴答声",
    "Rooster": "公鸡打鸣",
    "Dog": "狗叫声",
}


def display_class(label: str) -> str:
    return DISPLAY_LABELS.get(label, CLASS_LABELS.get(label, label))


def display_yamnet_label(label: str) -> str:
    return YAMNET_LABELS.get(label, label)


@st.cache_data(show_spinner=False)
def run_inference(
    audio_bytes: bytes, suffix: str, model_name: str
) -> tuple[dict, list[dict]]:
    """Cache YAMNet and classification results for the same file and model."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(audio_bytes)
        audio_path = Path(handle.name)
    try:
        result = analyze_audio(audio_path)
        predictions = predict_features(
            result["waveform"],
            result["sample_rate"],
            result["embeddings"],
            model_name,
        )
        return result, predictions
    finally:
        audio_path.unlink(missing_ok=True)


def section_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-overline">{number}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def waveform_figure(result: dict) -> plt.Figure:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(8, 3.1), dpi=130)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    timeline = np.arange(len(result["waveform"])) / result["sample_rate"]
    axis.plot(timeline, result["waveform"], linewidth=0.6, color="#087f72")
    axis.fill_between(timeline, result["waveform"], 0, color="#087f72", alpha=0.11)
    axis.set_xlabel("时间（秒）", color="#667b7d")
    axis.set_ylabel("振幅", color="#667b7d")
    axis.tick_params(colors="#667b7d", labelsize=8)
    axis.grid(axis="y", color="#e1e9e7", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color("#d7e2df")
    figure.tight_layout()
    return figure


def spectrogram_figure(result: dict) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 3.1), dpi=130)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    axis.imshow(result["spectrogram"].T, aspect="auto", origin="lower", cmap="magma")
    axis.set_xlabel("帧序号", color="#667b7d")
    axis.set_ylabel("梅尔频带", color="#667b7d")
    axis.tick_params(colors="#667b7d", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#d7e2df")
    figure.tight_layout()
    return figure


st.set_page_config(
    page_title="回声感知 | 环境声音识别",
    page_icon="sound",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17272a;
        --muted: #667b7d;
        --line: #d7e2df;
        --canvas: #f4f8f7;
        --paper: #ffffff;
        --mint: #087f72;
        --mint-soft: #e3f3ef;
        --coral: #c85f3e;
    }
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "Noto Sans SC", "Segoe UI", Arial, sans-serif;
        letter-spacing: 0;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container {
        max-width: 1160px;
        padding: 1.1rem 2.6rem 4rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stHorizontalBlock"] { gap: 1rem; }
    [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 2.2rem;
        margin-bottom: 0.65rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 750;
    }
    .brand-mark {
        display: grid;
        width: 1.7rem;
        height: 1.7rem;
        place-items: center;
        border: 1px solid var(--mint);
        color: var(--mint);
        font-size: 0.95rem;
    }
    .brand-sub {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 500;
    }
    .online {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: var(--mint);
        font-size: 0.78rem;
        font-weight: 700;
    }
    .online-dot {
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 50%;
        background: #2aa987;
    }
    .intro {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        min-height: 6.4rem;
        padding: 1rem 1.45rem;
        border: 1px solid #cfddda;
        background: var(--paper);
    }
    .intro-kicker {
        margin-bottom: 0.3rem;
        color: var(--mint);
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .intro h1 {
        margin: 0;
        color: var(--ink);
        font-size: clamp(1.55rem, 3vw, 2.35rem);
        font-weight: 750;
        line-height: 1.12;
    }
    .intro h1 em {
        color: var(--coral);
        font-style: normal;
    }
    .intro p {
        max-width: 410px;
        margin: 0;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.55;
        text-align: right;
    }
    .section-header {
        display: grid;
        grid-template-columns: auto auto minmax(0, 1fr);
        column-gap: 0.75rem;
        align-items: baseline;
        min-height: 3.6rem;
        padding: 1rem 0 0.65rem;
        border-bottom: 1px solid var(--line);
    }
    .section-overline {
        grid-column: 1;
        grid-row: 1;
        margin: 0;
        color: var(--ink);
        font-size: 1.18rem;
        font-weight: 750;
        letter-spacing: 0;
        white-space: nowrap;
    }
    .section-header h2 {
        grid-column: 2;
        grid-row: 1;
        margin: 0;
        color: var(--ink);
        font-size: 1.18rem;
        font-weight: 750;
    }
    .section-header p {
        grid-column: 3;
        grid-row: 1;
        margin: 0;
        color: var(--muted);
        font-size: 0.76rem;
        white-space: nowrap;
    }
    .st-key-audio_prep,
    .st-key-model_prep {
        min-height: 0;
        min-width: 0;
        max-width: 100%;
        box-sizing: border-box;
        overflow: hidden;
        padding: 1.15rem 1.25rem;
        border: 1px solid var(--line);
        background: var(--paper);
    }
    .st-key-audio_prep > div,
    .st-key-model_prep > div {
        min-width: 0;
        max-width: 100%;
        box-sizing: border-box;
    }
    .st-key-audio_prep [data-testid="stFileUploader"],
    .st-key-model_prep [data-baseweb="select"],
    .st-key-model_prep [data-baseweb="select"] > div {
        width: 100%;
        min-width: 0;
        max-width: 100%;
        box-sizing: border-box;
    }
    .panel-label {
        margin-bottom: 0.5rem;
        color: var(--muted);
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .panel-title {
        margin: 0 0 0.3rem;
        color: var(--ink);
        font-size: 1.03rem;
        font-weight: 750;
    }
    .panel-copy {
        margin: 0 0 0.9rem;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }
    div[data-testid="stFileUploader"] {
        margin-top: 0.7rem;
        padding: 0.2rem;
        border: 1px dashed #78aaa1;
        border-radius: 0;
        background: #f8fbfa;
    }
    div[data-testid="stFileUploader"] section { padding: 0.55rem; }
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small { color: var(--muted); }
    div[data-baseweb="select"] > div {
        min-height: 2.45rem;
        border-color: #b5cbc6;
        border-radius: 0;
        background: var(--paper);
    }
    div[data-baseweb="select"] span { color: var(--ink); }
    .model-note {
        margin-top: 0.75rem;
        padding: 0.58rem 0.7rem;
        border-left: 2px solid var(--coral);
        background: #fff7f3;
        color: var(--muted);
        font-size: 0.74rem;
        line-height: 1.4;
    }
    .dataset-note {
        margin-top: 0.65rem;
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.45;
    }
    .dataset-note a {
        color: var(--mint);
        text-decoration: none;
    }
    .empty-state {
        margin-top: 0.7rem;
        padding: 0.65rem 0.75rem;
        border: 1px solid #e6d5aa;
        background: #fffaf0;
        color: #946e1b;
        font-size: 0.78rem;
    }
    div[data-testid="stAudio"] { margin: 0.8rem 0 0; }
    div[data-testid="stAudio"] audio { width: 100%; }
    .result-hero, .rank-panel, .chart-frame {
        border: 1px solid var(--line);
        background: var(--paper);
    }
    .result-hero {
        padding: 1.45rem;
        border-color: #b7d8d1;
        background: var(--mint-soft);
    }
    .result-label {
        color: var(--mint);
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.08em;
    }
    .result-value {
        margin: 0.65rem 0 0.25rem;
        color: var(--ink);
        font-size: clamp(1.85rem, 3.2vw, 2.8rem);
        font-weight: 750;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }
    .result-confidence {
        color: var(--coral);
        font-size: 0.95rem;
        font-weight: 750;
    }
    .result-detail {
        margin-top: 0.9rem;
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }
    .rank-panel { padding: 1.2rem; }
    .rank-heading {
        margin-bottom: 0.25rem;
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.08em;
    }
    .rank-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        min-height: 2.6rem;
        border-bottom: 1px solid #edf1f0;
    }
    .rank-row:last-child { border-bottom: 0; }
    .rank-number {
        display: grid;
        width: 1.3rem;
        height: 1.3rem;
        place-items: center;
        border: 1px solid #9bc8bf;
        color: var(--mint);
        font-size: 0.68rem;
        font-weight: 750;
    }
    .rank-name {
        min-width: 6rem;
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 650;
    }
    .rank-track {
        flex: 1;
        height: 0.28rem;
        overflow: hidden;
        background: #e8efed;
    }
    .rank-fill {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, var(--mint), var(--coral));
    }
    .rank-score {
        width: 3rem;
        color: var(--muted);
        font-size: 0.74rem;
        text-align: right;
    }
    div[data-testid="stMetric"] {
        min-height: 5.3rem;
        padding: 0.85rem 0.95rem;
        border: 1px solid var(--line);
        border-radius: 0;
        background: var(--paper);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.68rem; }
    div[data-testid="stMetricValue"] { color: var(--ink); font-size: 1.2rem; }
    .chart-frame { padding: 0.8rem 0.9rem 0.35rem; }
    .chart-title { color: var(--ink); font-size: 0.86rem; font-weight: 750; }
    .chart-caption { margin: 0.15rem 0; color: var(--muted); font-size: 0.72rem; }
    div[data-testid="stTabs"] button { color: var(--muted); font-size: 0.8rem; font-weight: 700; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--mint); }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--mint); }
    .stAlert { border-radius: 0; }
    @media (max-width: 800px) {
        .block-container { padding: 0.9rem 1rem 3rem; }
        .intro { display: block; min-height: auto; padding: 1.1rem; }
        .intro p { margin-top: 0.5rem; text-align: left; }
        .st-key-audio_prep,
        .st-key-model_prep { min-height: 0; }
        .section-header {
            grid-template-columns: auto auto;
            row-gap: 0.2rem;
        }
        .section-header p {
            grid-column: 1 / -1;
            grid-row: 2;
            white-space: normal;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

available_models = [
    name
    for name in MODEL_NAMES
    if (MODEL_DIR / f"esc50_{name}.joblib").exists()
    and (MODEL_DIR / "esc50_labels.json").exists()
]
if not available_models:
    st.error("没有找到环境声音分类模型，请先完成特征提取和模型训练。")
    st.stop()

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <span class="brand-mark">∿</span>
            <span>回声感知 <span class="brand-sub">/ 环境声音识别</span></span>
        </div>
        <div class="online"><span class="online-dot"></span>模型在线</div>
    </div>
    <div class="intro">
        <div>
            <div class="intro-kicker">ESC-50 · 环境声音智能识别</div>
            <h1>把声音变成 <em>可读的信号。</em></h1>
        </div>
        <p>上传环境音，系统会自动识别，并在下方展示候选类别、波形和频谱。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

section_header("01 / 03", "准备分析", "上传音频，并选择用于识别的模型。")
setup_left, setup_right = st.columns([1.05, 1.15], gap="large")
with setup_left:
    with st.container(border=True, key="audio_prep"):
        st.markdown(
            """
            <div class="panel-label">音频输入</div>
            <div class="panel-title">选择一段声音</div>
            <p class="panel-copy">支持 WAV、FLAC、OGG 和 MP3，分析前统一重采样到 16 kHz。</p>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "音频文件",
            type=["wav", "flac", "ogg", "mp3"],
            label_visibility="collapsed",
        )

with setup_right:
    with st.container(border=True, key="model_prep"):
        st.markdown(
            """
            <div class="panel-label">分类器</div>
            <div class="panel-title">选择识别模型</div>
            <p class="panel-copy">默认使用 YAMNet 嵌入 + 支持向量机，可识别 ESC-50 的 50 类环境声音。</p>
            """,
            unsafe_allow_html=True,
        )
        model_name = st.selectbox(
            "分类器",
            available_models,
            index=available_models.index("yamnet_svm")
            if "yamnet_svm" in available_models
            else 0,
            format_func=lambda name: MODEL_LABELS[name],
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div class="model-note">{MODEL_LABELS[model_name]} 在独立测试集上的准确率为 {MODEL_ACCURACY[model_name]}，指标来自 ESC-50 官方测试折。</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="dataset-note">训练数据集：ESC-50 官方数据集。需要复现实验时，请从 <a href="https://github.com/karolpiczak/ESC-50" target="_blank">官方仓库</a> 下载；部署和使用网站不需要下载数据集。</div>',
            unsafe_allow_html=True,
        )

if uploaded is None:
    st.markdown(
        '<div class="empty-state">请先上传音频。上传后会自动开始识别，并在下方展示结果。</div>',
        unsafe_allow_html=True,
    )
    st.stop()

audio_bytes = uploaded.getvalue()
st.audio(audio_bytes, format=uploaded.type or "audio/wav")

with st.spinner("正在分析音频，请稍候…"):
    try:
        result, predictions = run_inference(
            audio_bytes,
            Path(uploaded.name).suffix,
            model_name,
        )
    except Exception as error:
        st.error(f"无法分析该音频文件：{error}")
        st.stop()

section_header("02 / 03", "识别结果", "查看最可能的声音类别和候选结果。")
result_left, result_right = st.columns([1.05, 1.15], gap="large")
with result_left:
    st.markdown(
        f"""
        <div class="result-hero">
            <div class="result-label">最可能的声音</div>
            <div class="result-value">{display_class(predictions[0]["label"])}</div>
            <div class="result-confidence">置信度 {predictions[0]["score"]:.1%}</div>
            <div class="result-detail">
                使用 {MODEL_LABELS[model_name]} 完成识别，结果来自 ESC-50 五十类环境声音标签。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with result_right:
    rank_rows = [
        '<div class="rank-panel"><div class="rank-heading">最接近的三个类别</div>'
    ]
    for rank, item in enumerate(predictions, start=1):
        width = max(4, int(item["score"] * 100))
        rank_rows.append(
            f'<div class="rank-row"><span class="rank-number">{rank}</span>'
            f'<span class="rank-name">{display_class(item["label"])}</span>'
            f'<span class="rank-track"><span class="rank-fill" style="width:{width}%"></span></span>'
            f'<span class="rank-score">{item["score"]:.1%}</span></div>'
        )
    rank_rows.append("</div>")
    st.markdown("".join(rank_rows), unsafe_allow_html=True)

duration, rate, embedding = st.columns(3)
duration.metric("音频时长", f"{result['duration']:.2f} 秒")
rate.metric("处理后采样率", f"{result['sample_rate'] / 1000:.0f} kHz")
embedding.metric("声音嵌入维度", f"{result['embeddings'].shape[1]} 维")

section_header("03 / 03", "声学分析", "从波形、频谱和 YAMNet 事件得分观察声音特征。")
waveform_tab, event_tab = st.tabs(["波形与频谱", "YAMNet 通用事件"])
with waveform_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="chart-frame"><div class="chart-title">音频波形</div><div class="chart-caption">时间域振幅变化</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(waveform_figure(result), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="chart-frame"><div class="chart-title">对数梅尔频谱图</div><div class="chart-caption">声音能量在频带中的分布</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(spectrogram_figure(result), clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)
with event_tab:
    st.caption("YAMNet 通用事件用于辅助观察；上方结果来自 ESC-50 五十类分类模型。")
    event_data = pd.DataFrame(result["top_results"])
    event_data["label"] = event_data["label"].map(display_yamnet_label)
    event_data = event_data[["label", "score"]].set_index("label")
    st.bar_chart(event_data, horizontal=True, x_label="YAMNet 得分")
