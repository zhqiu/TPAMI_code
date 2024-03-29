from train import SimCLR
import yaml
from dataloader.dataset_wrapper import DataSetWrapper
import torch
import argparse
import torch.distributed as dist

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.set_device(6)

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', default=3, type=int,
    #                     help='node rank for distributed training')
    # args = parser.parse_args()

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)

    main()
