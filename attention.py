import torch
import seaborn as sns


def plot_attentions(attentions, key_indices, tokenized_input, query_index=None, query_average=False, head_average=True, ax=None):
    if head_average:
        # attentions.shape = [num_layers, num_heads, input_length, input_length]
        selected_attentions = attentions.mean(dim=1) # selected_attentions.shape = [num_layers, input_length, input_length]
        selected_attentions = selected_attentions[:, :, key_indices] # selected_attentions.shape = [num_layers, input_length, len(key_indices)]
    else:
        # attentions.shape = [num_layers, input_length, input_length]
        selected_attentions = attentions[:, :, key_indices] # selected_attentions.shape = [num_layers, input_length, len(key_indices)]

    if query_index is None:
        selected_attentions = selected_attentions.mean(dim=1).softmax(dim=-1)
    else:
        selected_attentions = selected_attentions[:, query_index]
    
    selected_attentions = torch.flip(selected_attentions, dims=[0])
    x_tick_labels = [tokenized_input[i] for i in key_indices]
    num_layers = selected_attentions.shape[0]
    y_tick_labels = [str(i) if i % 2 == 0 else '' for i in range(num_layers, 0, -1)]
    sns.heatmap(selected_attentions.cpu(), xticklabels=x_tick_labels, yticklabels=y_tick_labels, cmap="YlOrRd", ax=ax)

    
def compute_rollout_attentions(residual_attentions):
    num_layers = residual_attentions.shape[0]
    rollout_attentions = torch.zeros_like(residual_attentions)
    rollout_attentions[0] = residual_attentions[0]
    for layer_index in range(1, num_layers):
        rollout_attentions[layer_index] = torch.matmul(residual_attentions[layer_index], rollout_attentions[layer_index-1])
    return rollout_attentions
