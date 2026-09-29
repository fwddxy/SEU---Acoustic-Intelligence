# ESC-50 环境声音识别系统

本项目使用预训练 YAMNet 提取音频表征，并在 ESC-50 全部 50 类环境声音上训练轻量分类器。系统提供中文 Streamlit 页面，可上传音频并查看分类结果、候选得分、波形、频谱和 YAMNet 通用事件。

项目同时保留 ESC-10 十分类实验，作为传统基线和对比实验，不再作为当前页面的主识别范围。

## 当前模型

页面默认使用 **YAMNet 嵌入 + 支持向量机** 的 ESC-50 五十分类模型。三个页面选项对应：

- YAMNet + 支持向量机
- YAMNet + 多层感知机
- MFCC + 支持向量机

ESC-50 模型的独立测试结果如下：

| 模型 | 识别类别数 | 测试样本 | 测试准确率 | 宏平均 F1 |
|---|---:|---:|---:|---:|
| YAMNet + 支持向量机 | 50 | 400 | 83.00% | 82.83% |
| YAMNet + 多层感知机 | 50 | 400 | 81.25% | 81.02% |
| MFCC + 支持向量机 | 50 | 400 | 46.50% | 44.43% |

测试集使用 ESC-50 官方 fold 5。训练使用 fold 1-3，验证使用 fold 4；验证折中与测试折来源编号重复的 2 条音频没有参与最终训练，因此最终训练样本为 1598 条，测试样本为 400 条。

上述准确率只代表 ESC-50 分布内的独立测试结果。用户上传的录音可能存在背景噪声、录音距离、响度、混响和类别不在 ESC-50 中等情况，因此不能保证任意音频都识别正确。模型概率是候选类别的相对得分，不等于现实世界中的绝对正确概率。

## 数据集下载

本仓库不上传 ESC-50 原始数据集。需要复现实验或重新训练模型时，请从 ESC-50 官方仓库下载：

- 官方仓库：[karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
- 项目脚本也支持自动下载，直接运行下面的命令即可：

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.prepare_data --all-classes
```

部署后的网页只使用已经训练好的模型，普通用户上传音频进行识别时不需要下载数据集。

## 运行环境

已验证的 Python 解释器：

```text
D:\Programs\miniconda\envs\Lecture2env\python.exe
```

依赖版本记录在 `requirements.txt` 中。首次运行 YAMNet 时会从 TensorFlow Hub 加载模型并缓存到本地。

## 从头训练 ESC-50 五十分类模型

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.prepare_data --no-download --all-classes
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.extract_features --manifest outputs\esc50_manifest.csv --output outputs\esc50_features.npz
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.train_esc50
```

如果本机还没有 ESC-50 数据集，去掉 `--no-download` 即可自动下载。特征提取每处理五个文件保存一次断点，意外中断后可以继续运行。

主要产物：

- 50 类数据清单：`outputs/esc50_manifest.csv`
- 50 类特征缓存：`outputs/esc50_features.npz`
- 50 类模型：`models/esc50_*.joblib`
- 50 类标签映射：`models/esc50_labels.json`
- 50 类实验报告：`outputs/esc50/summary.json`
- 50 类逐条预测和混淆矩阵：`outputs/esc50`

## 启动识别页面

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m streamlit run .\app.py
```

上传 WAV、FLAC、OGG 或 MP3 文件后，页面会显示：

- ESC-50 五十分类的最可能类别
- Top 3 候选类别及相对得分
- 音频时长、处理后采样率和嵌入维度
- 音频波形
- 对数梅尔频谱图
- YAMNet 通用声音事件得分

## 命令行识别

对已有音频运行当前模型：

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.predict .\data\ESC-50-master\audio\1-17970-A-4.wav
```

运行基础 YAMNet 检查：

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' .\run_demo.py
```

## ESC-10 对比实验

旧的 ESC-10 十分类实验仍保留，用于报告中的特征与分类器对比：

| 模型 | 测试准确率 | 宏平均 F1 |
|---|---:|---:|
| MFCC + 支持向量机 | 66.25% | 64.43% |
| YAMNet + 支持向量机 | 93.75% | 93.60% |
| YAMNet + 多层感知机 | 88.75% | 87.61% |

这些十分类指标与当前五十分类指标不是同一个任务，不能直接比较高低。

十分类实验文件位于：

- 特征缓存：`outputs/esc10_features.npz`
- 旧模型：`models/mfcc_svm.joblib`、`models/yamnet_svm.joblib`、`models/yamnet_mlp.joblib`
- 指标：`outputs/metrics`
- 混淆矩阵：`outputs/figures`
- 测试预测：`outputs/predictions`
