from paddlenlp.transformers import (BertModel, BertForTokenClassification)

import paddle
from paddle.nn import Layer, Linear



class PoetryBertModel(Layer):
    def __init__(self, pretrained_bert_model: str, input_length: int):
        super(PoetryBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_bert_model)
        self.vocab_size, self.hidden_size = self.bert.embeddings.word_embeddings.weight.shape
        self.classifier = Linear(self.hidden_size, self.vocab_size)

        self.sequence_length = input_length
        self.lower_triangle_mask = paddle.tril(paddle.ones((input_length, input_length), dtype='float32'))

    def forward(self, token, token_type, input_mask, input_length=None):
        mask_left = paddle.reshape(input_mask, input_mask.shape + [1])
        mask_right = paddle.reshape(input_mask, [input_mask.shape[0], 1, input_mask.shape[1]])
        mask_left = paddle.cast(mask_left, 'float32')
        mask_right = paddle.cast(mask_right, 'float32')
        attention_mask = paddle.matmul(mask_left, mask_right)

        if input_length is not None:
            lower_triangle_mask = paddle.tril(paddle.ones((input_length, input_length), dtype='float32'))
        else:
            lower_triangle_mask = self.lower_triangle_mask

        attention_mask = attention_mask * lower_triangle_mask
        attention_mask = (1 - paddle.unsqueeze(attention_mask, axis=[1])) * -1e10
        attention_mask = paddle.cast(attention_mask, self.bert.parameters()[0].dtype)

        # 得到BERT输出
        bert_output, _ = self.bert(
            input_ids=token,
            token_type_ids=token_type,
            attention_mask=attention_mask
        )

        # 输出 logits，形状：[batch_size, seq_len, vocab_size]
        logits = self.classifier(bert_output)

        return logits
# 定义模型损失
class PoetryBertModelLossCriterion(Layer):
    def forward(self, pred_logits, label, input_mask):
        loss = paddle.nn.functional.cross_entropy(pred_logits, label, ignore_index=0, reduction='none')
        masked_loss = paddle.mean(loss * input_mask, axis=0)
        return paddle.sum(masked_loss)