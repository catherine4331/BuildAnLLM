import torch

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

# Calculate attention scores for x^2
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# Normalise attention scores to get attention weights
attn_weights_2_tmp = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_tmp)

# But it's safer to use the torch method, as it is optimised and more stable
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# Calculate the context vector
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    # We linearly combine all our input vectors using their attention weights
    context_vec_2 += attn_weights_2[i] * x_i

# Now let's do this for real, calculating all the context vectors
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# But that is slow, let's do it properly
attn_scores = inputs @ inputs.T

# And normalise those
attn_weights = torch.softmax(attn_scores, dim=-1)

# Finally calculate all the context vectors
context_vecs = attn_weights @ inputs
print(context_vecs)
