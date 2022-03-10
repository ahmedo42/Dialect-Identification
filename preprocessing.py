import argparse
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from preprocessing import preprocess


class DialectIDModel(pl.LightningModule):
    def __init__(self, num_training_steps=0):
        super().__init__()
        n_classes = 18
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "UBC-NLP/MARBERT", num_labels=n_classes
        )
        self.num_training_steps = num_training_steps
        self.train_score = F1(num_classes=n_classes, average="macro")
        self.val_score = F1(num_classes=n_classes, average="macro")
        self.train_score.to(self.device)
        self.val_score.to(self.device)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = F.cross_entropy(preds, labels)
        self.log("train_loss", loss)

        self.train_score(preds, labels)
        self.log(
            "train_score", self.train_score, on_step=True, on_epoch=False, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = F.cross_entropy(preds, labels)
        self.log("val_loss", loss)

        self.val_score(preds, labels)
        self.log(
            "val_score", self.val_score, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class MARBERTDataset(Dataset):
    def __init__(self, fname, max_seq_len, test=False):
        super().__init__()
        self.df = pd.read_csv(fname, lineterminator="\n")
        self.tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
        self.max_seq_len = max_seq_len
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = preprocess(self.df.loc[idx, "text"], bert=True)
        encoded_input = self.tokenizer.encode_plus(
            text,
            padding="max_length",
            max_length=self.max_seq_len,
            add_special_tokens=True,
            truncation="longest_first",
        )

        if self.test:
            return torch.tensor(encoded_input["input_ids"]), torch.tensor(
                encoded_input["attention_mask"]
            )

        else:
            label = self.df.loc[idx, "dialect"]
            return (
                torch.tensor(encoded_input["input_ids"]),
                torch.tensor(encoded_input["attention_mask"]),
                torch.tensor(label, dtype=torch.int64),
            )


def main(args):
    print(args)
    pl.seed_everything(args.seed, workers=True)
    train_dataset = MARBERTDataset("train.csv", args.max_seq_len)
    val_dataset = MARBERTDataset("validation.csv", args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_training_steps = len(train_loader) * args.num_epochs
    model = DialectIDModel(num_training_steps)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_score", dirpath="./checkpoint/", verbose=True, mode="max"
    )
    callbacks = [
        checkpoint_callback,
        pl.callbacks.ProgressBar(refresh_rate=20),
    ]
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        accelerator="auto",
        check_val_every_n_epoch=1,
        gradient_clip_val=args.grad_clip,
        gpus=args.gpus,
        tpu_cores=args.tpu_cores,
    )
    trainer.fit(model, train_loader, val_loader)
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--grad_clip", default=None)
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--tpu_cores", default=None)

    args = parser.parse_args()
    main(args)
