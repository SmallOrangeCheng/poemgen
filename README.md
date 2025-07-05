# 🏯 AI 古诗生成系统

本项目基于 PaddleNLP 的中文 BERT 模型，训练并实现了一个支持藏头诗、风格控制、Top-K / Top-P / 温度采样的古典诗歌自动生成系统，并通过 Gradio 提供可视化界面。

---

## 📦 项目结构
├── dataprocess.py # 数据加载与预处理  
├── datareader.py # 自定义数据集类 PoemData  
├── model.py # 模型结构定义（PoetryBertModel + Loss）  
├── poemgenerate.py # 自动生成诗歌逻辑 PoetryGen  
├── train.py # 训练主程序入口  
├── turn.py # 微调模型  
├── ui.py # Gradio 可视化界面（输入汉字生成诗）  
├── checkpoint/ # 模型保存路径  
├── requirements/ # 依赖文件  
└── README.md # 当前说明文档


---

## 🚀 快速开始

### 1️⃣ 安装依赖

请使用 Python ≥ 3.8，并安装以下依赖：

```bash
pip install -r requirements.txt
```

requirements.txt 内容如下：  
paddlepaddle==3.0.0         # 安装时请根据是否有 GPU 选择对应版本  
paddlenlp==2.8.1  
numpy>=1.19  
gradio>=4.0.0  
torch>=1.10.0               



### 2️⃣ 模型训练
运行 train.py 脚本开始训练模型：
```bash
python train.py
```
训练使用的主要参数如下：

| 参数名             | 含义说明                           | 默认值         |
|------------------|----------------------------------|--------------|
| `epochs`         | 训练轮数                           | 10           |
| `batch_size`     | 批次大小                           | 128 (train)，32 (dev) |
| `max_len`        | 每句诗的最大长度                    | 128          |
| `learning_rate`  | 学习率（适用于 BERT 微调）           | 0.0001       |
| `save_dir`       | 模型保存目录                         | `./checkpoint` |
| `save_freq`      | 每多少轮保存一次模型                  | 1            |
| `eval_freq`      | 每多少轮进行一次评估                  | 1            |

### 3️⃣ 启动可视化生成界面
运行 Gradio 界面：
```bash
python ui.py
```
然后在浏览器访问：http://127.0.0.1:7862

你将看到一个中文界面，输入一个汉字，选择风格，即可生成古诗。

## 📝 功能介绍
🌟 支持“藏头诗”生成（可输入一个字或一个字列表）

🌈 可选择“五言绝句”“七言绝句”两种格式

🎯 提供 Top-K、Top-P、温度采样方式控制生成多样性

🖥️ 网页端支持复制诗歌文本

🎨 示例 

| 输入汉字 | 风格   | 示例输出                             |
| ---- | ---- | -------------------------------- |
| 春    | 五言绝句 | 春山烟雨近，陌上草青青。雁落天边影，风来柳自轻。         |
| 月    | 七言绝句 | 月明千里照孤舟，江上寒潮夜不休。谁念长安旧梦事，愁心如水共潮流。 |

 