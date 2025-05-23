import webdataset as wds
import torch
import numpy as np
import pickle
from tqdm import tqdm
import heapq
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from SAE import SparseAutoencoder

block_code = sys.argv[1]
num_neurons = int(sys.argv[2])
k = int(sys.argv[3])
checkpoint_prefix = sys.argv[4]
suffix = sys.argv[5]
save_previews = False

code_to_block = {
    "down": "unet.down_blocks.2.attentions.1",
    "mid": "unet.mid_block.attentions.0",
    "up": "unet.up_blocks.0.attentions.1",
    "up0": "unet.up_blocks.0.attentions.0",
}

block_name = code_to_block[block_code]
path = "/share/datasets/datasets/sdxl-turbo-latents/2025-05-06 12:18:50.608813/"
dataset = wds.WebDataset(f"{path}{block_name}.tar").decode("torch")
cutoff = 2000

num_top_images = 100

sae_name = f'{block_name}_k{k}_hidden{num_neurons}_auxk256_bs4096_lr0.0001'

path_to_checkpoint = f"{checkpoint_prefix}/{sae_name}/{suffix}"

sae = SparseAutoencoder.load_from_disk(
    path_to_checkpoint
).to('cuda')

images = []

topk_values = torch.full((num_top_images, num_neurons), float('-inf')).cuda()
topk_indices = torch.full((num_top_images, num_neurons), -1).cuda()
topk_heatmaps = torch.full((num_top_images, 16, 16, num_neurons), -1).cuda()

class RunningStats:
    def __init__(self):
        self.n = 0.0

        self.n_activated = torch.zeros(num_neurons).cuda()
        self.activation_rate = torch.zeros(num_neurons).cuda()
        self.mean = torch.zeros(num_neurons).cuda()
        self.var = torch.zeros(num_neurons).cuda()
    
    def update_activated(self, new_tensor):
        act_tensor = (new_tensor > 0)
        self.n_activated += act_tensor.sum(dim=0)
        self.activation_rate = self.n_activated / (self.n + new_tensor.shape[0])

    def update_mean_var(self, new_tensor):
        # Update mean and var for each neuron if activated (value > 0)
        act_tensor = (new_tensor > 0)

        batch_n_activated = act_tensor.sum(dim=0)

        batch_n_activated = torch.maximum(batch_n_activated, torch.ones_like(batch_n_activated) * 1e-9)
        batch_mean = torch.sum(new_tensor, dim=0) / batch_n_activated
        batch_var = torch.sum(((new_tensor - batch_mean) * act_tensor) ** 2, dim=0) / batch_n_activated

        # Update mean and var for each neuron
        new_mean = (self.n_activated * self.mean + batch_n_activated * batch_mean) / (self.n_activated + batch_n_activated)
        new_var = (self.var * self.n_activated + batch_var * batch_n_activated) / (self.n_activated + batch_n_activated) + \
                (self.n_activated * batch_n_activated / (self.n_activated + batch_n_activated).pow(2)) * (batch_mean - self.mean).pow(2)
        new_n_activated = self.n_activated + batch_n_activated

        self.mean = new_mean
        self.var = new_var
        self.n_activated = new_n_activated

    def update(self, new_tensor):
        # new_tensor: (batch_size, 1, width, height, num_neurons)
        assert new_tensor.shape[-1] == num_neurons
        new_tensor = new_tensor.reshape(-1, num_neurons)

        self.update_activated(new_tensor)
        self.update_mean_var(new_tensor)

        self.n += new_tensor.shape[0]

    def get_mean_std_act_rate(self):
        std = torch.sqrt(self.var)
        return self.mean, std, self.activation_rate

running_stats = RunningStats()

def update_topk(new_tensor, current_topk_values, current_topk_indices, current_topk_heatmaps, batch_offset):
    new_indices = torch.arange(batch_offset, batch_offset + new_tensor.shape[0]).unsqueeze(1).expand(-1, num_neurons).to('cuda')
    new_values = new_tensor.mean(dim=(1, 2, 3))
    new_heatmaps = new_tensor.mean(dim=(1))

    all_values = torch.cat([current_topk_values, new_values], dim=0)
    all_indices = torch.cat([current_topk_indices, new_indices], dim=0)
    all_heatmaps = torch.cat([current_topk_heatmaps, new_heatmaps], dim=0)

    # all_values.shape = (num_images, num_neurons)
    # all_indices.shape = (num_images, num_neurons)
    # all_heatmaps.shape = (num_images, width, height, num_neurons)

    topk_values, topk_indices = torch.topk(all_values, num_top_images, dim=0, largest=True, sorted=True)
    topk_heatmaps = torch.gather(all_heatmaps, 0, topk_indices.unsqueeze(1).unsqueeze(2).expand(-1, all_heatmaps.shape[1], all_heatmaps.shape[2], -1))
    topk_indices = torch.gather(all_indices, 0, topk_indices)

    return topk_values, topk_indices, topk_heatmaps

batch_offset = 0

for i, sample in tqdm(enumerate(dataset)):
    # tensor = sample["output.pth"].permute((0, 1, 3, 4, 2))
    diff = sample["diff.pth"].permute((0, 1, 3, 4, 2))
    #print(diff.shape)
    with torch.no_grad():
        acts = sae.encode(diff.to('cuda'))
    #print(acts.shape)
    running_stats.update(acts)

    topk_values, topk_indices, topk_heatmaps = update_topk(acts, topk_values, topk_indices, topk_heatmaps, batch_offset)
    batch_offset += diff.shape[0]

    if i == cutoff - 1:
        break
    
torch.save(topk_values, f'{path_to_checkpoint}/topk_vals.pt')
torch.save(topk_indices, f'{path_to_checkpoint}/topk_indices.pt')
torch.save(topk_heatmaps, f'{path_to_checkpoint}/topk_heatmaps.pt')

mean, std, act_rate = running_stats.get_mean_std_act_rate()
torch.save(mean, f'{path_to_checkpoint}/mean.pt')
torch.save(std, f'{path_to_checkpoint}/std.pt')
torch.save(act_rate, f'{path_to_checkpoint}/act_rate.pt')

if save_previews:

    dataset_images = wds.WebDataset(f"{path}images.tar").decode('torch')
    for i, sample in tqdm(enumerate(dataset_images)):
        images.append(sample['images.npy'])
        if i == cutoff:
            break

    images = np.concatenate(images, axis=0)
    from tqdm import tqdm
    from PIL import Image
    from p_tqdm import p_uimap

    # Function to process each feat_idx
    def process_feat_idx(feat_idx):
        os.makedirs(f"resourses/{sae_name}/imgs/{feat_idx:06d}", exist_ok=True)
        for i in range(topk_indices.shape[0]):
            # try:
                img = Image.fromarray(images[topk_indices[i, feat_idx].item()])
                img = img.resize((128, 128))  # Resize the image to 128x128
                img.save(f"resourses/{sae_name}/imgs/{feat_idx:06d}/{i:06d}.png")
            # except Exception as e:
            #     print(e)
            #     continue

    topk_indices = topk_indices.cpu()

    # # List of feat_idx to process
    feat_idx_list = range(topk_indices.shape[1])

    # # Use p_uimap to process feat_idx in parallel
    list(p_uimap(process_feat_idx, feat_idx_list))
    # for feat_idx in tqdm(feat_idx_list):
    #     process_feat_idx(feat_idx)

print("Processing completed.")