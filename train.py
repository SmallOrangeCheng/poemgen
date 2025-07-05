
from dataprocess import train_dataset, bert_tokenizer, dev_dataset
from datareader import PoemData
from tune import model  # 你定义的 paddle.Model 实例

from paddle.io import DataLoader

import matplotlib.pyplot as plt


from paddle.callbacks import Callback

class TrainAndEvalPPLCallback(Callback):
    def __init__(self, train_loader, eval_loader, model):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epochs = []
        self.train_ppls = []
        self.eval_ppls = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)

        self.model.network.eval()
        train_result = self.model.evaluate(self.train_loader, verbose=0)
        train_ppl = train_result.get("Perplexity", None)
        if train_ppl is not None:
            self.train_ppls.append(train_ppl)
            print(f"[Train] Epoch {epoch + 1}: Train PPL = {train_ppl:.2f}")

        eval_result = self.model.evaluate(self.eval_loader, verbose=0)
        eval_ppl = eval_result.get("Perplexity", None)
        if eval_ppl is not None:
            self.eval_ppls.append(eval_ppl)
            print(f"[Eval ] Epoch {epoch + 1}: Eval  PPL = {eval_ppl:.2f}")

        self.model.network.train()

    def on_train_end(self, logs=None):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_ppls, label="Train PPL", marker='o', color='blue')
        plt.plot(self.epochs, self.eval_ppls, label="Eval PPL", marker='o', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.title("Perplexity on Train and Eval Dataset")
        plt.legend()
        plt.grid(True)
        plt.savefig("ppl_per_epoch.png")
        print("[Callback] Saved PPL curve to ppl_per_epoch.png")







train_loader = DataLoader(PoemData(train_dataset, bert_tokenizer, 128), batch_size=128, shuffle=True)
dev_loader = DataLoader(PoemData(dev_dataset, bert_tokenizer, 128), batch_size=18, shuffle=True)

# 创建回调列表（一定要放在 model 和 train_loader 定义之后）
callback_list = [TrainAndEvalPPLCallback(train_loader, dev_loader, model)]

model.fit(train_data=train_loader, epochs=10, save_dir='./checkpoint', save_freq=1, verbose=1, eval_data=dev_loader, eval_freq=1,callbacks=callback_list)


