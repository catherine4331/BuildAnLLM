import torch
import torch.nn as nn
import tiktoken

from transformer.gpt import GPTModel, GPT_CONFIG_MEDIUM

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx

start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_MEDIUM)
model.eval()

out = generate_text_simple(
    model, encoded_tensor, 6, GPT_CONFIG_MEDIUM["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)