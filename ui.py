import gradio as gr
import torch
from test import *



from dataprocess import bert_tokenizer
from model import PoetryBertModel
import paddle
import re

from poemgenerate import PoetryGen

# 载入已经训练好的模型
net = PoetryBertModel('bert-base-chinese', 128)
model = paddle.Model(net)
model.load('./checkpoint/final')
poetry_gen = PoetryGen(model, bert_tokenizer)

def poetry_show(poetry):
    pattern = r"([，。；？])"
    text = re.sub(pattern, r'\1 ', poetry)
    for p in text.split():
        if p:
            print(p)


def generate_chinese_poem(input_char,style):
    """
    根据输入汉字和选择的风格生成诗歌
    """
    print(input_char)
    if len(input_char) != 1:
        return "⚠️ 请只输入一个汉字", "生成结果为空，无法复制"
    if not '\u4e00' <= input_char <= '\u9fff':
        return "⚠️ 请输入有效的汉字", "生成结果为空，无法复制"

    # 这里替换为你的实际模型调用代码
    poem = poetry_gen.generate(head=input_char,type = style)
    print(poem)
    return poem, poem  # 返回两次poem，一次用于显示，一次用于复制




description_md = """
# AI 古诗生成器 🎭
请输入一个汉字，AI将为您创作一首以该汉字开头的诗歌。
"""

content_md = """
---
## 📝 详细说明

### 功能介绍
本项目使用 PyTorch 和LSTM实现古诗生成模型，可以根据用户输入的单个汉字，生成对应的古典诗歌。

### 使用说明
1. 在输入框中输入**一个汉字**
2. 选择想要的**诗歌风格**
3. 点击"生成诗歌"按钮
4. 可以点击"复制诗歌"按钮复制生成的内容

### 创作技巧
- 可以尝试输入季节词语（如：春、夏、秋、冬）
- 可以使用表达情感的字词（如：愁、思、忆）
- 建议选择意境优美的汉字

### 支持的诗歌格式
- 五言绝句
- 七言绝句



"""

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(description_md)

    with gr.Row():
        input_char = gr.Textbox(
            lines=1,
            placeholder="请输入一个汉字...",
            label="输入汉字"
        )
        style = gr.Radio(
            choices=["五言绝句", "七言绝句"],
            label="诗歌风格",
            value="五言绝句"
        )

    # 创建一个隐藏的文本框用于复制功能
    copy_text = gr.Textbox(visible=False)

    # 显示诗歌的文本框
    output = gr.Textbox(
        lines=4,
        label="生成的诗歌"
    )

    # 按钮行
    with gr.Row():
        generate_btn = gr.Button("生成诗歌", variant="primary")
        copy_btn = gr.Button("复制诗歌")

    # 示例区域
    gr.Examples(
        examples=[
            ["春", "五言绝句"],
            ["月", "七言绝句"],

        ],
        inputs=[input_char, style],
        outputs=[output, copy_text]
    )

    # 添加详细说明
    gr.Markdown(content_md)

    # 设置按钮功能
    generate_btn.click(
        fn=generate_chinese_poem,
        inputs=[input_char, style],
        outputs=[output, copy_text]
    )

    # 添加复制功能
    copy_btn.click(
        None,
        copy_text,
        None,
        js="""
        (text) => {
            if (text === "" || text === "生成结果为空，无法复制") {
                alert("请先生成诗歌！");
                return;
            }
            navigator.clipboard.writeText(text);
            alert("诗歌已复制到剪贴板！");
        }
        """
    )

# 启动服务
if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7862, share=False)
