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
compute_previews = False

code_to_block_help = {
    "down": "unet.down_blocks.2.attentions.1",
    "mid": "unet.mid_block.attentions.0",
    "up": "unet.up_blocks.0.attentions.1",
    "up0": "unet.up_blocks.0.attentions.0",
}

code_to_path = {
    "down": "/share/datasets/datasets/sdxl-turbo-neurons-down/2025-05-06 22:14:14.656654/",
    "mid": "/share/datasets/datasets/sdxl-turbo-neurons-mid/2025-05-07 01:13:59.542656/",
    "up0": "/share/datasets/datasets/sdxl-turbo-neurons-up/2025-05-07 04:15:08.449822/",
    "up": "/share/datasets/datasets/sdxl-turbo-neurons-up1/2025-05-07 07:17:54.942466/",
}

path = code_to_path[block_code.split(".")[0]]

code_to_block = {}
for key, block in code_to_block_help.items():
    for i in range(10):
        code_to_block[f"{key}.{i}"] = f"{block}.transformer_blocks.{i}.ff.net.0"


block_name = code_to_block[block_code]
# path = "/share/datasets/datasets/sdxl-turbo-neurons/2025-05-06 16:52:16.279418/"
#path = "/share/datasets/datasets/sdxl-turbo-neurons-down/2025-05-06 22:14:14.656654/"
dataset = wds.WebDataset(f"{path}{block_name}.tar").decode("torch")
cutoff = 2000 # 5000

num_top_images = 10

images = []

topk_values = torch.full((num_top_images, num_neurons), float('-inf'))
topk_indices = torch.full((num_top_images, num_neurons), -1, dtype=torch.long)
topk_heatmaps = torch.full((num_top_images, 16, 16, num_neurons), -1)

class RunningStats:
    def __init__(self):
        self.n = 0.0

        self.n_activated = torch.zeros(num_neurons)
        self.activation_rate = torch.zeros(num_neurons)
        self.mean = torch.zeros(num_neurons)
        self.var = torch.zeros(num_neurons)
    
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
    new_indices = torch.arange(batch_offset, batch_offset + new_tensor.shape[0]).unsqueeze(1).expand(-1, num_neurons)
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

    acts = sample["neurons_first512.pth"]
    acts = acts.reshape(acts.shape[0], acts.shape[1], 16, 16, acts.shape[3])
    running_stats.update(acts)

    topk_values, topk_indices, topk_heatmaps = update_topk(acts, topk_values, topk_indices, topk_heatmaps, batch_offset)
    batch_offset += acts.shape[0]

    if i == cutoff - 1:
        break


os.makedirs(f"resourses/{block_name}", exist_ok=True)
    
torch.save(topk_values, f'resourses/{block_name}/topk_vals.pt')
torch.save(topk_indices, f'resourses/{block_name}/topk_indices.pt')
torch.save(topk_heatmaps, f'resourses/{block_name}/topk_heatmaps.pt')

mean, std, act_rate = running_stats.get_mean_std_act_rate()
torch.save(mean, f'resourses/{block_name}/mean.pt')
torch.save(std, f'resourses/{block_name}/std.pt')
torch.save(act_rate, f'resourses/{block_name}/act_rate.pt')

if compute_previews:
    from tqdm import tqdm
    from PIL import Image
    from p_tqdm import p_uimap

    dataset_images = wds.WebDataset(f"{path}images.tar").decode('torch')
    for i, sample in tqdm(enumerate(dataset_images)):
        images.append(sample['images.npy'])

        if i == cutoff - 1:
            break

    images = np.concatenate(images, axis=0)

    # Function to process each feat_idx
    def process_feat_idx(feat_idx):
        os.makedirs(f"resourses/{block_name}/imgs/{feat_idx:06d}", exist_ok=True)
        for i in range(topk_indices.shape[0]):
            # try:
                img = Image.fromarray(images[topk_indices[i, feat_idx].item()])
                img = img.resize((128, 128))  # Resize the image to 128x128
                img.save(f"resourses/{block_name}/imgs/{feat_idx:06d}/{i:06d}.png")
            # except Exception as e:
            #     print(e)
            #     continue

    # List of feat_idx to process
    feat_idx_list = range(topk_indices.shape[1])

    # Use p_uimap to process feat_idx in parallel
    list(p_uimap(process_feat_idx, feat_idx_list))
    #for feat_idx in tqdm(feat_idx_list):
    #     process_feat_idx(feat_idx)

print("Processing completed.")