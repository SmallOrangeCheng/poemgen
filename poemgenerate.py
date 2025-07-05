
import numpy as np
import paddle

class PoetryGen(object):
    """
    定义一个自动生成诗句的类，按照要求生成诗句
    model: 训练得到的预测模型
    tokenizer: 分词编码工具
    max_length: 生成诗句的最大长度，需小于等于model所允许的最大长度
    """
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.puncs = ['，', '。', '？', '；']
        self.max_length = max_length

    def generate(self, style='', head='', type='', topk=0, temperature=1.0, topp=0.0):
        if '五言绝句' in type:
            words_per_line = 5
        elif '七言绝句' in type:
            words_per_line = 7
        else:
            words_per_line = 7
        num_lines = 4

        style_ids = self.tokenizer.encode(style)['input_ids'][:-1]
        head_index = 0
        head_is_list = isinstance(head, list)
        if head_is_list:
            current_line = self.tokenizer.encode(head[head_index])['input_ids'][1:-1]
            head_index += 1
        else:
            current_line = self.tokenizer.encode(head)['input_ids'][1:-1]

        poetry_lines = []
        for line_num in range(num_lines):
            while len(current_line) < words_per_line:
                next_word = self._gen_next_word(style_ids + sum(poetry_lines, []) + current_line, topk, temperature,
                                                topp)
                if next_word in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]', '[SEP]']):
                    continue
                if next_word in self.tokenizer.convert_tokens_to_ids(self.puncs):
                    continue
                current_line.append(next_word)

            # 添加标点
            if line_num < num_lines - 1:
                current_line.append(self.tokenizer.convert_tokens_to_ids(['，'])[0])
            else:
                current_line.append(self.tokenizer.convert_tokens_to_ids(['。'])[0])
            poetry_lines.append(current_line)
            current_line = []

            # 如果是藏头诗
            if head_is_list and head_index < len(head):
                current_line = self.tokenizer.encode(head[head_index])['input_ids'][1:-1]
                head_index += 1

        all_ids = [token for line in poetry_lines for token in line]
        return ''.join(self.tokenizer.convert_ids_to_tokens(all_ids))

    def _gen_next_word(self, known_ids, topk=0, temperature=1.0, topp=0.0):
        """
        生成下一个词的ID，支持Top-K，温度，Top-P采样
        topk=0 不启用Top-K
        topp=0 不启用Top-P
        temperature=1.0 不调整概率分布
        """
        type_token = [0] * len(known_ids)
        mask = [1] * len(known_ids)
        sequence_length = len(known_ids)
        known_ids = paddle.to_tensor([known_ids], dtype='int64')
        type_token = paddle.to_tensor([type_token], dtype='int64')
        mask = paddle.to_tensor([mask], dtype='float32')
        logits = self.model.network.forward(known_ids, type_token, mask, sequence_length)
        logits = logits[0, -1, :].numpy()

        # 温度调整
        if temperature != 1.0:
            logits = logits / temperature

        # 先转成概率分布
        exp_logits = np.exp(logits - np.max(logits))  # 减去最大值防止溢出
        probs = exp_logits / exp_logits.sum()

        # Top-P (核采样)
        if topp > 0.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            # 只保留累计概率小于 topp 的词
            cutoff = cumulative_probs <= topp
            # 确保至少保留一个词
            if not np.any(cutoff):
                cutoff[0] = True
            filtered_indices = sorted_indices[cutoff]
            filtered_probs = sorted_probs[cutoff]
            filtered_probs /= filtered_probs.sum()
            word_chosen = np.random.choice(filtered_indices, p=filtered_probs)
            return word_chosen

        # Top-K 采样
        if topk > 0:
            topk_indices = probs.argsort()[::-1][:topk]
            topk_probs = probs[topk_indices]
            topk_probs /= topk_probs.sum()
            word_chosen = np.random.choice(topk_indices, p=topk_probs)
            return word_chosen

        # 不用 Top-K 或 Top-P，直接根据全部词概率采样
        word_chosen = np.random.choice(len(probs), p=probs)
        return word_chosen

