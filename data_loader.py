import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from transformers import AutoTokenizer

def make_fixed_length(array, max_length, padding_value=0):
    if len(array) >= max_length:
        return array[:max_length]
    else:
        pad_length = max_length - len(array)
        return array + [padding_value] * pad_length

def construct_dataset(data, tokenizer, max_seq_length=512):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    targets = []
    for doc in data:
        targets.append(doc["target"])
        # Run tokenizer on input text and unpack into different data fields.
        binarized = tokenizer(doc["text"])
        input_ids.append(make_fixed_length(binarized["input_ids"], max_seq_length))
        token_type_ids.append(make_fixed_length(binarized["token_type_ids"], max_seq_length))
        attention_mask.append(make_fixed_length(binarized["attention_mask"], max_seq_length))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, targets)
    return dataset

class DrugSynergyDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data, tokenizer, train_batch_size=32, dev_batch_size=32, test_batch_size=32, dev_train_ratio=0.1, max_seq_length=512):
        '''
        dev_train_ratio: hold out x% of the training set as a dev set.
        '''
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.test_batch_size = test_batch_size
        self.dev_train_ratio = dev_train_ratio
        self.max_seq_length = max_seq_length

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # TODO(Vijay): set dimensions here
        self.dims = (1, 28, 28)

    def setup(self):
        # Assign train/val datasets for use in dataloaders
        full_dataset = construct_dataset(self.train_data, self.tokenizer, max_seq_length=self.max_seq_length)
        dev_size = int(self.dev_train_ratio * len(full_dataset))    
        train_size = len(full_dataset) - dev_size
        self.train, self.val = random_split(full_dataset, [train_size, dev_size])
        self.test = construct_dataset(self.test_data, self.tokenizer, max_seq_length=self.max_seq_length)
        # Optionally...
        # self.dims = tuple(self.train[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.dev_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size)