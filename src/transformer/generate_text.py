import torch
import torch.nn as nn
import tiktoken

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

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# start_context = "Hello, I am"
# tokenizer = tiktoken.get_encoding("gpt2")
# encoded = tokenizer.encode(start_context)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_MEDIUM)
# model.eval()

# out = generate_text_simple(
#     model, encoded_tensor, 6, GPT_CONFIG_MEDIUM["context_length"]
# )
# decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)

# torch.manual_seed(123)
# token_ids = generate(
#     model,
#     text_to_token_ids("Every effort moves you", tokenizer),
#     15,
#     GPT_CONFIG_MEDIUM["context_length"],
#     25,
#     1.4
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))