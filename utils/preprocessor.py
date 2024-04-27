import json
import os
import sys

import torch

import pandas as pd

from transformers import AutoTokenizer
from indobenchmark import IndoNLGTokenizer

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
import lightning as L

class Preprocessor(L.LightningDataModule):
    def __init__(self, max_length, batch_size, lm_model = None):
        self.indosum_dir = "datasets/indosum"

        if lm_model:
            # selain indobart
            self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        else:
            # indobart
            self.tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
        
        self.max_length = max_length
        self.batch_size = batch_size
    
    def join_paragraphs(self, paragraphs):
        # Join List of paragraphs to string paragraphs
        string_paragraph = ""
        for parag in paragraphs:
            for kal in parag:
                kalimat = " ".join(kal)
                string_paragraph += kalimat 
        
        return string_paragraph
    
    def join_summary(self, summaries):
        string_summary = ""
        # Join Kalimat dalam Satu Paragraf
        for sumr in summaries:
            kal_sum = " ".join(sumr)
            string_summary += kal_sum

        return string_summary
            
    def load_data(self, flag):
        list_files = os.listdir(self.indosum_dir)
        datasets = []
        for fl in list_files:
            if flag in fl:
                with open(f"{self.indosum_dir}/{fl}", "r", encoding = "utf-8") as json_reader:
                    # load file jsonl (jsonl = kumpulan file json format di gabung jadi satu file)
                    
                    data_raw = json_reader.readlines()                   
                    # json_raw = [json.loads(jline) for jline in json_reader.readlines().rstrip().splitlines()]
                
                json_raw = []  
                for dd in data_raw:
                    
                    data_line = json.loads(dd)
                    paragraphs = self.join_paragraphs(data_line["paragraphs"])
                    summary = self.join_summary(data_line["summary"])
                    
                    data = {
                        "id": data_line["id"],
                        "paragraphs": paragraphs,
                        "summary": summary,
                    }
                    
    
                    json_raw.append(data)
                
                datasets += json_raw
        
        # print(len(datasets))
        
        return datasets
    
    
    def list2tensor(self, data):
        x_ids, x_att, y_ids, y_att = [], [], [], []
        for i_d, d in enumerate(tqdm(data)):
            # Token untuk Paragraphs (X)
            x_tok = self.tokenizer(
                d["paragraphs"], truncation = True,
                max_length = self.max_length,
                padding = "max_length"
            )

            # Token untuk summary (Y)
            y_tok = self.tokenizer(
                d["summary"],
                truncation = True,
                max_length= self.max_length,
                padding= "max_length"
            )

            x_ids.append(x_tok["input_ids"])
            x_att.append(x_tok["attention_mask"])
            y_ids.append(y_tok["input_ids"])
            y_att.append(y_tok["attention_mask"])

            # if i_d > 100:
            #     break


        x_ids = torch.tensor(x_ids)
        x_att = torch.tensor(x_att)

        y_ids = torch.tensor(y_ids)
        y_att = torch.tensor(y_att)

        return TensorDataset(x_ids, x_att, y_ids, y_att)
        # x_ids = token
        # x_att = attention_mask


    def preprocessor(self):
        if not os.path.exists("datasets/preprocessed/indosum/train.pt"):
            # raw data masih dalam format list
            raw_train_data = self.load_data(flag = "train")
            raw_val_data = self.load_data(flag = "dev")
            raw_test_data = self.load_data(flag = "test")

            # list => tensor dataset
            train_data = self.list2tensor(data = raw_train_data)
            val_data = self.list2tensor(raw_val_data)
            test_data = self.list2tensor(raw_test_data)

            # save
            torch.save(train_data, "datasets/preprocessed/indosum/train.pt")
            torch.save(val_data, "datasets/preprocessed/indosum/val.pt")
            torch.save(test_data, "datasets/preprocessed/indosum/test.pt")
        else:
            # load existing preprocessed data
            train_data = torch.load(train_data, "datasets/preprocessed/indosum/train.pt")
            val_data = torch.load(val_data, "datasets/preprocessed/indosum/val.pt")
            test_data = torch.load(test_data, "datasets/preprocessed/indosum/test.pt")

        return train_data, val_data, test_data
    

    def setup(self, stage = None):
        train_data, val_data, test_data = self.preprocessor()

        if stage == "fit":
            self.train_data = train_data
            self.val_data = val_data
        elif stage == "test":
            self.test_data = test_data


    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 3
        )

    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 3
        )

    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 3
        )

if __name__ == "__main__":
    pre = Preprocessor(max_length = 512, batch_size = 5, )
    pre.setup(stage = "fit")
    for data in pre.train_dataloader():
        print(len(data))
        sys.exit()