import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from transformers import BartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer

import evaluate


class IndoSum(L.LightningModule):
    def __init__(
        self,
    ):
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
        self.rouge = evaluate.load("rouge")

    # x = data yang belum di summarize
    # y = data label
    def forward(self, x_ids, x_att, y_ids):
        return self.model(input_ids=x_ids, attention_mask=x_att, labels=y_ids)

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
