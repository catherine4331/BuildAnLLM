from pathlib import Path
import tiktoken
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformer.gpt import GPT_CONFIG_124M, GPTModel
from openai.pretrained import download_and_load_gpt2, load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from transformer.generate_text import generate_text_simple

class SpamDataset(Dataset):
    def __init__(self, csv_file: Path, tokenizer: tiktoken.Encoding, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
val_dataset = SpamDataset(
    csv_file="validation.csv",
     max_length=None,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
     max_length=None,
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

settings, params = download_and_load_gpt2(
    model_size = "124M", models_dir="gpt2"
)

model = GPTModel(NEW_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model,
    text_to_token_ids(text_1, tokenizer).to(device),
    15,
    NEW_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

text_2 = ("Is the following text 'spam'? Answer with a 'yes' or 'no':"
         " 'You are a winner you have been specially"
         " selected to receive $1000 cash or a $2000 award.'"
         )
token_ids = generate_text_simple(
    model,
    text_to_token_ids(text_2, tokenizer).to(device),
    23,
    NEW_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))