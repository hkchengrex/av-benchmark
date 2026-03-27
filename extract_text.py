import os
import logging
from argparse import ArgumentParser
from pathlib import Path

import laion_clap
import pandas as pd
import torch
from colorlog import ColoredFormatter
from msclap import CLAP
from tqdm import tqdm

from av_bench.utils import resolve_ckpt

home_dir = os.path.expanduser("~")
_clap_ckpt_path = Path(home_dir) / '.cache' / 'av-benchmark' / 'weights' / 'music_speech_audioset_epoch_15_esc_89.98.pt'
log = logging.getLogger()
device = 'cuda'

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


setup_eval_logging()


@torch.inference_mode()
def extract(args):
    text_csv = args.text_csv
    output_cache_path = args.output_cache_path

    output_cache_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(text_csv).to_dict(orient='records')

    laion_clap_model = laion_clap.CLAP_Module(enable_fusion=False,
                                              amodel='HTSAT-base').cuda().eval()
    laion_clap_model.load_ckpt(resolve_ckpt(_clap_ckpt_path), verbose=False)

    ms_clap_model = CLAP(version='2023', use_cuda=True)

    all_laion_clap = {}
    all_ms_clap = {}
    for row in tqdm(df):
        name = str(row['name'])
        caption = row['caption']

        text_data = [caption]
        text_embed = laion_clap_model.get_text_embedding(text_data, use_tensor=True)
        all_laion_clap[name] = text_embed.cpu().squeeze()

        text_embed = ms_clap_model.get_text_embeddings(text_data)
        all_ms_clap[name] = text_embed.cpu().squeeze()

    torch.save(all_laion_clap, output_cache_path / 'clap_laion_text.pth')
    torch.save(all_ms_clap, output_cache_path / 'clap_ms_text.pth')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--text_csv', type=Path, required=True)
    parser.add_argument('--output_cache_path', type=Path, required=True)
    args = parser.parse_args()
    extract(args)
