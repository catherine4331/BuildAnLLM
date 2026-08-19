import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# Initialise our weight matrices
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

# We need all the key and value vectors to calculate the context vector for x^2
keys = inputs @ W_key
values = inputs @ W_value

# The attention score for each token is the dot product of the query vector with the key vector
# Note these are the transformed input vectors that have gone through out weight matrices
attn_scores_2 = query_2 @ keys.T

# Now we calculate the attention weights using the softmax function. We divide by the root of d_k 
# as this is the variance of the dot product of Q and K (it grows larger with the key embedding dimension)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)

# Now we calculate the context vector, as a weighted sum of each value vector
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)