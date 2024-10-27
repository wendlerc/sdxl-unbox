import os
import sys
import io
import tarfile
import torch
import webdataset as wds
import numpy as np

from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from SDLens.hooked_sd_pipeline import HookedStableDiffusionXLPipeline

import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
import diffusers
import fire

def main(save_path, start_at=0, finish_at=30000, dataset_batch_size=50):
    blocks_to_save = [
        'unet.down_blocks.2.attentions.1',
        'unet.mid_block.attentions.0',
        'unet.up_blocks.0.attentions.0',
        'unet.up_blocks.0.attentions.1',
    ]

    # Initialization
    dataset = load_dataset("guangyil/laion-coco-aesthetic", split="train", columns=["caption"], streaming=True).shuffle(seed=42)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained('stabilityai/sdxl-turbo')
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    dataloader = DataLoader(dataset, batch_size=dataset_batch_size)

    ct = datetime.datetime.now()
    save_path = os.path.join(save_path, str(ct))
    # Collecting dataset
    os.makedirs(save_path, exist_ok=True)

    writers = {
        block: wds.TarWriter(f'{save_path}/{block}.tar') for block in blocks_to_save
    }

    writers.update({'images': wds.TarWriter(f'{save_path}/images.tar')})

    def to_kwargs(kwargs_to_save):
        kwargs = kwargs_to_save.copy()
        seed = kwargs['seed']
        del kwargs['seed']
        kwargs['generator'] = torch.Generator(device="cpu").manual_seed(num_document)
        return kwargs

    dataloader_iter = iter(dataloader)
    for num_document, batch in tqdm(enumerate(dataloader)):
        if num_document < start_at:
            continue

        if num_document >= finish_at:
            break

        kwargs_to_save = {
            'prompt': batch['caption'],
            'positions_to_cache': blocks_to_save,
            'save_input': True,
            'save_output': True,
            'num_inference_steps': 1,
            'guidance_scale': 0.0,
            'seed': num_document,
            'output_type': 'pil'
        }

        kwargs = to_kwargs(kwargs_to_save)

        output, cache = pipe.run_with_cache(
            **kwargs
        )

        blocks = cache['input'].keys()
        for block in blocks:
            sample = {
                "__key__": f"sample_{num_document}",
                "output.pth": cache['output'][block],
                "diff.pth": cache['output'][block] - cache['input'][block],
                "gen_args.json": kwargs_to_save
            }

            writers[block].write(sample)
        writers['images'].write({
            "__key__": f"sample_{num_document}",
            "images.npy": np.stack(output.images)
        })

    for block, writer in writers.items():
        writer.close()

if __name__ == '__main__':
    fire.Fire(main)