from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.functional import accuracy

from datamodule import tokenize_text
from config import Config, DataModuleConfig, ModuleConfig


class SequenceClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        num_classes: int = DataModuleConfig.num_classes,  # set according to the finetuning dataset
        input_key: str = "input_ids",  # set according to the finetuning dataset
        label_key: str = "label",  # set according to the finetuning dataset
        mask_key: str = "attention_mask",  # set according to the model output object
        output_key: str = "logits",  # set according to the model output object
        loss_key: str = "loss",  # set according to the model output object
        learning_rate: float = ModuleConfig.learning_rate,
    ) -> None:
        """a custom LightningModule for sequence classification

        Args:
            model_name: the name of the model and accompanying tokenizer
            num_classes: number of classes
            input_key: key used to access token input ids
            label_key: key used to access labels of model output
            mask_key: key used to access attention mask of model output
            output_key: key used to access prediction tensor
            loss_key: key used to access model return output
            learning_rate: learning rate passed to optimizer
        """
        super().__init__()

        self.model_name = model_name
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.accuracy = accuracy
        self.input_key = input_key
        self.label_key = label_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss_key = loss_key

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training
        """
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        self.log("train-loss", outputs[self.loss_key])
        return outputs[self.loss_key]

    def validation_step(self, batch, batch_idx) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
        """
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )
        self.log("val-loss", outputs[self.loss_key], prog_bar=True)

        logits = outputs[self.output_key]
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels,
            batch["label"],
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("val-acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#testing
        """
        outputs = self.model(
            batch[self.input_key],
            attention_mask=batch[self.mask_key],
            labels=batch[self.label_key],
        )

        logits = outputs[self.output_key]
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels,
            batch["label"],
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("test-acc", acc, prog_bar=True)

    def predict_step(
        self, sequence: str, cache_dir: Union[str, Path] = Config.cache_dir
    ) -> str:
        """
        Notes:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#inference
        """
        batch = tokenize_text(sequence, model_name=self.model_name, cache_dir=cache_dir)
        # autotokenizer may cause tokens to lose device type and cause failure
        batch = batch.to(self.device)
        outputs = self.model(batch[self.input_key])
        logits = outputs[self.output_key]
        predicted_label_id = torch.argmax(logits)
        labels = {0: "negative", 1: "positive"}
        return labels[predicted_label_id.item()]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
