# ESC-10 环境声音识别系统

本项目基于 ESC-50 数据集中的 ESC-10 子集，完成环境声音十分类实验，比较以下三种方法：

- MFCC + 支持向量机
- YAMNet 声音嵌入 + 支持向量机
- YAMNet 声音嵌入 + 多层感知机

项目同时提供中文 Streamlit 识别页面，可以上传音频并查看分类结果、置信度、波形和频谱图。

## 运行环境

已验证的 Python 解释器为：

```text
D:\Programs\miniconda\envs\Lecture2env\python.exe
```

依赖版本记录在 `requirements.txt` 中。首次运行时，YAMNet 会从 TensorFlow Hub 加载并缓存到本地。
以下命令均在项目目录的 PowerShell 中执行。

## 复现实验

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.prepare_data --no-download
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.extract_features
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.train
```

数据准备命令默认使用已经解压到 `data/ESC-50-master` 的数据集。新环境中没有数据集时，去掉 `--no-download` 即可自动下载。
特征提取每处理五个文件保存一次断点，意外中断后可以继续运行。

数据划分遵循 ESC-50 官方 fold：fold 1-3 为训练集，共 240 条；fold 4 为验证集，共 80 条；fold 5 为测试集，共 80 条。
当前分类器只使用训练集训练，并在独立测试集上报告结果，验证集暂时保留用于后续调参。

模型推理增加了开放集拒识机制：当分类器置信度低于验证集确定的阈值时，不再强行输出一个 ESC-10 类别；如果 YAMNet 的通用 521 类声音事件中存在明确结果，则优先给出该通用声音提示。例如，青蛙声不属于 ESC-10 十类，系统会提示青蛙声，而不是误报成时钟滴答声。

当前实验结果：

| 方法 | 测试准确率 | 宏平均 F1 |
|---|---:|---:|
| MFCC + 支持向量机 | 66.25% | 64.43% |
| YAMNet + 支持向量机 | 93.75% | 93.60% |
| YAMNet + 多层感知机 | 88.75% | 87.61% |

实验产物位置：

- 特征缓存：`outputs/esc10_features.npz`
- 训练模型：`models/*.joblib`
- 评价指标：`outputs/metrics`
- 混淆矩阵：`outputs/figures`
- 测试集预测明细：`outputs/predictions`

## 启动中文识别页面

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m streamlit run .\app.py
```

页面默认使用 YAMNet + 支持向量机，也可以切换另外两种分类器。上传音频后，页面会显示：

- ESC-10 最可能类别和置信度
- Top 3 候选类别及得分条
- 音频时长、处理后采样率和嵌入维度
- 音频波形
- 对数梅尔频谱图
- YAMNet 原始声音事件得分

支持的音频格式为 WAV、FLAC、OGG 和 MP3。

## 命令行识别

对已有音频运行三种 ESC-10 分类器：

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' -m src.predict .\data\ESC-50-master\audio\5-170338-A-41.wav
```

运行基础 YAMNet 检查：

```powershell
& 'D:\Programs\miniconda\envs\Lecture2env\python.exe' .\run_demo.py
```
