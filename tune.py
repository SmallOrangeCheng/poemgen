# 针对预训练模型的训练，需使用较小的学习率（learning_rate）进行调优。
import paddle
from model import PoetryBertModel, PoetryBertModelLossCriterion

from paddle.static import InputSpec
from paddlenlp.metrics import Perplexity
from paddle.optimizer import AdamW

net = PoetryBertModel('bert-base-chinese', 128)

token_ids = InputSpec((-1, 128), 'int64', 'token')
token_type_ids = InputSpec((-1, 128), 'int64', 'token_type')
input_mask = InputSpec((-1, 128), 'float32', 'input_mask')
label = InputSpec((-1, 128), 'int64', 'label')

inputs = [token_ids, token_type_ids, input_mask]
labels = [label, input_mask]

model = paddle.Model(net, inputs, labels)
model.prepare(optimizer=AdamW(learning_rate=0.0001, parameters=model.parameters()), loss=PoetryBertModelLossCriterion(),
              metrics=[Perplexity()])

model.summary(inputs, [input.dtype for input in inputs])
