import re
import torchmetrics
import torch

import pandas as pd
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification


class DialectIDModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        n_classes = 18
        self.model = AutoModelForSequenceClassification.from_pretrained("UBC-NLP/MARBERT",num_labels=n_classes)
        self.train_score = torchmetrics.F1Score(num_classes= n_classes,average="macro")
        self.val_score =  torchmetrics.F1Score(num_classes= n_classes,average="macro")
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = F.cross_entropy(preds, labels)
        self.log("train_loss", loss)
        
        self.train_score(preds, labels)
        self.log("train_score", self.train_score, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        preds = self(input_ids, attention_mask)
        loss = F.cross_entropy(preds, labels)
        self.log("val_loss", loss,prog_bar=True)
        
        self.val_score(preds, labels)
        self.log("val_score", self.val_score, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config["learning_rate"])



class MARBERTDataset(Dataset):
    def __init__(self, fname, max_seq_len, test=False):
        super().__init__()
        self.df = pd.read_csv(fname,lineterminator='\n')
        self.tokenizer =  AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
        self.max_seq_len = max_seq_len
        self.test = test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        text = preprocess(text,bert=True)
        encoded_input = self.tokenizer.encode_plus(text, padding='max_length', max_length=self.max_seq_len, 
                                                   add_special_tokens=True, truncation='longest_first')
        
        if self.test:
            return torch.tensor(encoded_input["input_ids"]), torch.tensor(encoded_input["attention_mask"])        
            
        else:
            label = self.df.loc[idx, "dialect"]
            return torch.tensor(encoded_input["input_ids"]), torch.tensor(encoded_input["attention_mask"]), torch.tensor(label, dtype=torch.int64)


if __name__ == "__main__":
    config = {
        'learning_rate': 2e-6,
        'max_seq_len': 128,
        'batch_size': 32,
        'num_workers': 2,
        'num_epochs': 5,
    }

    train_dataset = MARBERTDataset("train.csv", config["max_seq_len"])
    val_dataset = MARBERTDataset("validation.csv",config["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    model = DialectIDModel()
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val_score", dirpath='./checkpoint/', verbose=True, mode="max"),
    ]
    trainer = pl.Trainer(max_epochs=config['num_epochs'], callbacks=callbacks, gpus=1)    
    trainer.fit(model, train_loader, val_loader)