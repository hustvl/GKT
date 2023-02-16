from pathlib import Path
from tqdm import tqdm
import time
import os
import torch
import pytorch_lightning as pl
import hydra

from cross_view_transformer.common import setup_config, setup_network, setup_data_module

def setup(cfg):
    print('Benchmark mixed precision by adding +mixed_precision=True')
    print('Benchmark cpu performance +device=cpu')

    cfg.loader.batch_size = 1

    if 'mixed_precision' not in cfg:
        cfg.mixed_precision = False

    if 'device' not in cfg:
        cfg.device = 'cuda'


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    pl.seed_everything(2022, workers=True)

    network = setup_network(cfg)
    data = setup_data_module(cfg)
    loader = data.train_dataloader(shuffle=False)

    device = torch.device(cfg.device)

    network = network.to(device)
    network.eval()

    sample = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor)
             else v for k, v in sample.items()}
    with torch.no_grad():
        latency = []
        for _ in range(1050):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            network(batch)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency.append(end_time - start_time)
        latency = latency[50:]
    latency = torch.tensor(latency).mean().item()
    print("inference latency: {:.3f} ms, speed: {:.3f} fps".format(
        latency * 1000, 1.0 / latency
    ))


if __name__ == '__main__':
    main()
