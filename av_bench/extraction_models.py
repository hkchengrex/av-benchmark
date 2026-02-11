import os
from pathlib import Path

import laion_clap
import torch
import torch.nn as nn
import torchaudio
from hear21passt.base import get_basic_model
from imagebind.models import imagebind_model
from msclap import CLAP

from av_bench.panns import Cnn14
from av_bench.synchformer.synchformer import Synchformer
from av_bench.vggish.vggish import VGGish
from av_bench.utils import resolve_ckpt

home_dir = os.path.expanduser("~")
_clap_ckpt_path = Path(home_dir) / '.cache' / 'av-benchmark' / 'weights' / 'music_speech_audioset_epoch_15_esc_89.98.pt'
_syncformer_ckpt_path =  Path(home_dir) / '.cache' / 'av-benchmark' / 'weights' / 'synchformer_state_dict.pth'


class ExtractionModels(nn.Module):

    def __init__(self):
        super().__init__()

        features_list = ["2048", "logits"]
        self.panns = Cnn14(
            features_list=features_list,
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )

        self.panns = self.panns.eval()
        self.vggish = VGGish(postprocess=False).eval()

        # before the prediction head
        # https://github.com/kkoutini/passt_hear21/blob/5f1cce6a54b88faf0abad82ed428355e7931213a/hear21passt/models/passt.py#L440-L441
        self.passt_model = get_basic_model(mode="all")
        self.passt_model.eval()

        self.imagebind = imagebind_model.imagebind_huge(pretrained=True).eval()

        self.laion_clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base').eval()
        self.laion_clap.load_ckpt(resolve_ckpt(_clap_ckpt_path), verbose=False)

        self.ms_clap = CLAP(version='2023', use_cuda=True)

        self.synchformer = Synchformer().eval()
        sd = torch.load(resolve_ckpt(_syncformer_ckpt_path), weights_only=True)
        self.synchformer.load_state_dict(sd)

        # from synchformer
        self.sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            win_length=400,
            hop_length=160,
            n_fft=1024,
            n_mels=128,
        )
