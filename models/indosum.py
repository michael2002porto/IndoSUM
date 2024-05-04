import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from transformers import BartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer

import evaluate


class IndoSum(L.LightningModule):
    def __init__(self, lr):
        super(IndoSum, self).__init__()
        # method constructor
        # BART
        # input → kasih tau output nya seperti ini (encoder) → decoder (generate predicted output) → output
        # # input → encoder → decoder → output
        # GPT
        # input → decoder → output
        self.model = BartForConditionalGeneration.from_pretrained(
            "indobenchmark/indobart-v2"
        )
        self.tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")

        # rouge untuk benchmarking = scoring model
        # model → output (prediction) → bandingin dengan label (rouge)
        # x = output prediction, y = kalimat tersimpulkan
        # rouge (x, y) = 0.2

        # self.rouge = evaluate.load('rouge')
        self.rouge = ROUGEScore()

        self.train_step_output = []
        self.val_step_output = []
        self.test_step_output = []

        # self.save_hyperparameters()
        self.lr = lr

    # x = data yang belum di summarize
    # y = data label
    def forward(self, x_ids, x_att, y_ids):
        return self.model(input_ids=x_ids, attention_mask=x_att, labels=y_ids)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 = 0.0001

    def training_step(self, batch, batch_idx):
        x_ids, x_att, y_ids, y_att = batch

        # run model
        output_model = self(x_ids, x_att, y_ids)

        metrics = {}

        # self.loss(y_ids, output_model)

        metrics["loss"] = output_model.loss

        decode_output = output_model.logits.topk(1, dim=-1)[1]
        # logits = vector embedding output
        # topk = max & index
        # dim = -1 (horizontal)
        # dim = 1 (vertikal)

        decode_output - decode_output.squeeze().tolist()
        # squeeze = ambil value paling dalam dari 3 kurung siku [[[1, 2, 3]]] => [1, 2, 3]
        # tolist() = tensor dijadikan list

        decoded_pred = []
        decoded_true = []

        # tokenizer = ambil kata dari kalimat
        # tokenizer encode = ubah kata menjadi angka
        # tokenizer decode = ubah angka menjadi kata
        for dcd_x, dcd_y in zip(decode_output, y_ids):
            dec_pred = self.tokenizer.decode(dcd_x, skip_special_tokens=True)
            dec_true = self.tokenizer.decode(dcd_y, skip_special_tokens=True)

            decoded_pred.append(dec_pred)
            decoded_true.append(dec_true)

        # Benchmarking output model
        # rouge = benchmarking teknik untuk tahu hasil prediksi dengan summarize yang asli berapa persen
        # rouge = untuk mengetahui model kita bagus atau tidak
        try:
            # rouge_scores = self.rouge.compute(
            #     predictions = decoded_pred,
            #     reference = decoded_true,
            #     use_stemmer=True,
            #     tokenizer = self.tokenizer
            # )

            rouge_scores = self.rouge(preds=decoded_pred, target=decoded_true)

            metrics["train_rouge1"] = round(
                rouge_scores["rouge1_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rouge2"] = round(
                rouge_scores["rouge2_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rougeL"] = round(
                rouge_scores["rougeL_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rougeL_sum"] = round(
                rouge_scores["rougeLsum_fmeasure"].cpu().tolist(), 3
            )
        except ZeroDivisionError:
            metrics["train_rouge1"] = 0.0
            metrics["train_rouge2"] = 0.0
            metrics["train_rougeL"] = 0.0
            metrics["train_rougeL_sum"] = 0.001

        self.train_step_output.append(metrics)
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        x_ids, x_att, y_ids, y_att = batch

        # run model
        output_model = self(x_ids, x_att, y_ids)

        metrics = {}

        metrics["loss"] = output_model.loss

        # Ambil output model dan susun probability token maksimum
        decode_output = output_model.logits.topk(1, dim=-1)[1]

        decode_output = decode_output.squeeze().tolist()

        decoded_pred = []
        decoded_true = []

        for dcd_x, dcd_y in zip(decode_output, y_ids):
            dec_pred = self.tokenizer.decode(dcd_x, skip_special_tokens=True)
            dec_true = self.tokenizer.decode(dcd_y, skip_special_tokens=True)

            decoded_pred.append(dec_pred)
            decoded_true.append(dec_true)

        # Benchmarking output model
        try:
            # rouge_scores = self.rouge.compute(
            #     predictions = decoded_pred,
            #     reference = decoded_true,
            #     use_stemmer=True,
            #     tokenizer = self.tokenizer
            # )
            # metrics["train_rouge1"] = round(rouge_scores["rouge1"], 3)
            # metrics["train_rouge2"] = round(rouge_scores["rouge2"], 3)
            # metrics["train_rougeL"] = round(rouge_scores["rougeL"], 3)
            # metrics["train_rougeL_sum"] = round(rouge_scores["rougeLsum"], 3)
            rouge_scores = self.rouge(preds=decoded_pred, target=decoded_true)

            metrics["train_rouge1"] = round(
                rouge_scores["rouge1_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rouge2"] = round(
                rouge_scores["rouge2_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rougeL"] = round(
                rouge_scores["rougeL_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rougeL_sum"] = round(
                rouge_scores["rougeLsum_fmeasure"].cpu().tolist(), 3
            )

        except ZeroDivisionError:
            metrics["train_rouge1"] = 0.0
            metrics["train_rouge2"] = 0.0
            metrics["train_rougeL"] = 0.0
            metrics["train_rougeL_sum"] = 0.001

        self.val_step_output.append(metrics)
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return metrics["loss"]

    def test_step(self, batch, batch_idx):
        x_ids, x_att, y_ids, y_att = batch
        # run model
        output_model = self(x_ids, x_att, y_ids)

        metrics = {}

        metrics["loss"] = output_model.loss

        # Ambil output model dan susun probability token maksimum
        decode_output = output_model.logits.topk(1, dim=-1)[1]

        decode_output = decode_output.squeeze().tolist()

        decoded_pred = []
        decoded_true = []

        for dcd_x, dcd_y in zip(decode_output, y_ids):
            dec_pred = self.tokenizer.decode(dcd_x, skip_special_tokens=True)
            dec_true = self.tokenizer.decode(dcd_y, skip_special_tokens=True)

            decoded_pred.append(dec_pred)
            decoded_true.append(dec_true)

        # Benchmarking output model
        try:
            # rouge_scores = self.rouge.compute(
            #     predictions = decoded_pred,
            #     reference = decoded_true,
            #     use_stemmer=True,
            #     tokenizer = self.tokenizer
            # )
            # metrics["train_rouge1"] = round(rouge_scores["rouge1"], 3)
            # metrics["train_rouge2"] = round(rouge_scores["rouge2"], 3)
            # metrics["train_rougeL"] = round(rouge_scores["rougeL"], 3)
            # metrics["train_rougeL_sum"] = round(rouge_scores["rougeLsum"], 3)
            rouge_scores = self.rouge(preds=decoded_pred, target=decoded_true)

            metrics["train_rouge1"] = round(
                rouge_scores["rouge1_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rouge2"] = round(
                rouge_scores["rouge2_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rougeL"] = round(
                rouge_scores["rougeL_fmeasure"].cpu().tolist(), 3
            )
            metrics["train_rougeL_sum"] = round(
                rouge_scores["rougeLsum_fmeasure"].cpu().tolist(), 3
            )

        except ZeroDivisionError:
            metrics["train_rouge1"] = 0.0
            metrics["train_rouge2"] = 0.0
            metrics["train_rougeL"] = 0.0
            metrics["train_rougeL_sum"] = 0.001

        self.test_step_output.append(metrics)
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return metrics["loss"]
