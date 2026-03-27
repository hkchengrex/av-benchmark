import os
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from av_bench.metrics import compute_fd, compute_isc, compute_kl
from av_bench.synchformer.synchformer import Synchformer, make_class_grid
from av_bench.utils import (unroll_dict, unroll_dict_all_keys, unroll_paired_dict,
                            unroll_paired_dict_with_key, resolve_ckpt)

home_dir = os.path.expanduser("~")
_syncformer_ckpt_path =  Path(home_dir) / '.cache' / 'av-benchmark' / 'weights' / 'synchformer_state_dict.pth'
log = logging.getLogger()
device = 'cuda'


@torch.inference_mode()
def evaluate(gt_audio_cache: Path,
             pred_audio_cache: Path,
             *,
             is_paired: bool = True,
             num_samples: int = 1,
             skip_video_related: bool = False,
             skip_clap: bool = False,
             clean_sample_names: bool = False) -> Dict[str, float]:

    sync_model = Synchformer().to(device).eval()
    sd = torch.load(resolve_ckpt(_syncformer_ckpt_path), weights_only=True)
    sync_model.load_state_dict(sd)

    gt_audio_cache = gt_audio_cache.expanduser()
    pred_audio_cache = pred_audio_cache.expanduser()

    gt_pann_features = torch.load(gt_audio_cache / 'pann_features.pth', weights_only=True)
    pred_pann_features = torch.load(pred_audio_cache / 'pann_features.pth', weights_only=True)

    gt_vggish_features = torch.load(gt_audio_cache / 'vggish_features.pth', weights_only=True)
    pred_vggish_features = torch.load(pred_audio_cache / 'vggish_features.pth', weights_only=True)

    gt_passt_features = torch.load(gt_audio_cache / 'passt_features_embed.pth', weights_only=True)
    pred_passt_features = torch.load(pred_audio_cache / 'passt_features_embed.pth',
                                     weights_only=True)

    gt_passt_logits = torch.load(gt_audio_cache / 'passt_logits.pth', weights_only=True)
    pred_passt_logits = torch.load(pred_audio_cache / 'passt_logits.pth', weights_only=True)

    # convert these dictionaries (with filenames as keys) to lists
    if is_paired:
        paired_panns_logits, gt_panns_logits = unroll_paired_dict_with_key(gt_pann_features,
                                                                           pred_pann_features,
                                                                           num_samples=num_samples)

        if not skip_video_related and (gt_audio_cache / 'imagebind_video.pth').exists():
            ib_video_features = torch.load(gt_audio_cache / 'imagebind_video.pth',
                                           weights_only=True)
            ib_audio_features = torch.load(pred_audio_cache / 'imagebind_audio.pth',
                                           weights_only=True)
            paired_ib_video_features, paired_ib_audio_features, unpaired_ib_keys = unroll_paired_dict(
                ib_video_features, ib_audio_features, clean_sample_names=clean_sample_names)
            log.info(f'Unpaired IB features keys: {unpaired_ib_keys}')
        else:
            paired_ib_video_features = paired_ib_audio_features = None
            log.info('No IB features found, skipping IB-score evaluation')

        if not skip_video_related and (gt_audio_cache / 'synchformer_video.pth').exists():
            sync_video_features = torch.load(gt_audio_cache / 'synchformer_video.pth',
                                             weights_only=True)
            sync_audio_features = torch.load(pred_audio_cache / 'synchformer_audio.pth',
                                             weights_only=True)
            paired_sync_video_features, paired_sync_audio_features, unpaired_sync_keys = unroll_paired_dict(
                sync_video_features, sync_audio_features, clean_sample_names=clean_sample_names)
            log.info(f'Unpaired Synchformer features keys: {unpaired_sync_keys}')
        else:
            paired_sync_video_features = paired_sync_audio_features = None
            log.info('No Synchformer features found, skipping DeSync evaluation')

        if not skip_clap and (gt_audio_cache / 'clap_laion_text.pth').exists():
            laion_clap_text_features = torch.load(gt_audio_cache / 'clap_laion_text.pth',
                                                  weights_only=True)
            laion_clap_audio_features = torch.load(pred_audio_cache / 'clap_laion_audio.pth',
                                                   weights_only=True)
            paired_laion_clap_text_features, paired_laion_clap_audio_features, unpaired_laion_clap_keys = unroll_paired_dict(
                laion_clap_text_features, laion_clap_audio_features, clean_sample_names=clean_sample_names)
            log.info(f'Unpaired LAION CLAP features keys: {unpaired_laion_clap_keys}')

            ms_clap_text_features = torch.load(gt_audio_cache / 'clap_ms_text.pth',
                                               weights_only=True)
            ms_clap_audio_features = torch.load(pred_audio_cache / 'clap_ms_audio.pth',
                                                weights_only=True)
            paired_ms_clap_text_features, paired_ms_clap_audio_features, unpaired_ms_clap_keys = unroll_paired_dict(
                ms_clap_text_features, ms_clap_audio_features, clean_sample_names=clean_sample_names)
            log.info(f'Unpaired MS CLAP features keys: {unpaired_ms_clap_keys}')
        else:
            paired_laion_clap_text_features = paired_laion_clap_audio_features = None
            paired_ms_clap_text_features = paired_ms_clap_audio_features = None
            log.info('No CLAP features found, skipping CLAP-score evaluation')
    else:
        paired_panns_logits = gt_panns_logits = None
        paired_ib_video_features = paired_ib_audio_features = None

    gt_pann_features = unroll_dict_all_keys(gt_pann_features)
    pred_pann_features = unroll_dict_all_keys(pred_pann_features)

    if is_paired:
        gt_passt_features, pred_passt_features, unpaired_passt_keys = unroll_paired_dict(
            gt_passt_features, pred_passt_features, clean_sample_names=clean_sample_names)
        log.info(f'Unpaired PASST features keys: {unpaired_passt_keys}')

        gt_passt_logits, pred_passt_logits, unpaired_passt_keys = unroll_paired_dict(
            gt_passt_logits, pred_passt_logits, clean_sample_names=clean_sample_names)
        log.info(f'Unpaired PASST logits keys: {unpaired_passt_keys}')

    else:
        gt_passt_features = unroll_dict(gt_passt_features)
        pred_passt_features = unroll_dict(pred_passt_features)

        gt_passt_logits = unroll_dict(gt_passt_logits)
        pred_passt_logits = unroll_dict(pred_passt_logits)

    gt_vggish_features = unroll_dict(gt_vggish_features, cat=True)
    pred_vggish_features = unroll_dict(pred_vggish_features, cat=True)

    output_metrics = {}

    fd_vgg = compute_fd(pred_vggish_features.numpy(), gt_vggish_features.numpy())
    output_metrics['FD-VGG'] = fd_vgg

    # fd_pann = compute_fd(pred_pann_features['2048'].numpy(), gt_pann_features['2048'].numpy())
    fd_pann = compute_fd(pred_pann_features['2048'].numpy(), gt_pann_features['2048'].numpy())
    output_metrics['FD-PANN'] = fd_pann

    fd_passt = compute_fd(pred_passt_features.numpy(), gt_passt_features.numpy())
    output_metrics['FD-PASST'] = fd_passt

    if is_paired:
        # SpecVQGAN / Diff-Foley should be using the softmax version
        kl_metrics = compute_kl(paired_panns_logits, gt_panns_logits)
        output_metrics['KL-PANNS-softmax'] = kl_metrics['kl_softmax']

        kl_metrics = compute_kl([pred_passt_logits], gt_passt_logits)
        output_metrics['KL-PASST-softmax'] = kl_metrics['kl_softmax']

    metric_isc = compute_isc(
        pred_pann_features,
        feat_layer_name='logits',
        splits=10,
        samples_shuffle=True,
        rng_seed=2020,
    )
    output_metrics['ISC-PANNS-mean'] = metric_isc['inception_score_mean']
    output_metrics['ISC-PANNS-std'] = metric_isc['inception_score_std']

    metrics_isc = compute_isc(
        pred_passt_logits,
        feat_layer_name=None,
        splits=10,
        samples_shuffle=True,
        rng_seed=2020,
    )
    output_metrics['ISC-PASST-mean'] = metrics_isc['inception_score_mean']
    output_metrics['ISC-PASST-std'] = metrics_isc['inception_score_std']

    if is_paired and paired_ib_video_features is not None:
        # compute ib score
        ib_score = torch.cosine_similarity(paired_ib_video_features,
                                           paired_ib_audio_features,
                                           dim=-1).mean()
        output_metrics['IB-Score'] = ib_score.item()

    if is_paired and paired_sync_video_features is not None:
        # compute sync score
        batch_size = 16
        total_samples = paired_sync_video_features.shape[0]
        total_sync_scores = []
        sync_grid = make_class_grid(-2, 2, 21)
        for i in tqdm(range(0, total_samples, batch_size)):
            sync_video_batch = paired_sync_video_features[i:i + batch_size].to(device)
            sync_audio_batch = paired_sync_audio_features[i:i + batch_size].to(device)
            logits = sync_model.compare_v_a(sync_video_batch[:, :14], sync_audio_batch[:, :14])
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            for j in range(sync_video_batch.shape[0]):
                total_sync_scores.append(abs(sync_grid[top_id[j]].item()))

            logits = sync_model.compare_v_a(sync_video_batch[:, -14:], sync_audio_batch[:, -14:])
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            for j in range(sync_video_batch.shape[0]):
                total_sync_scores.append(abs(sync_grid[top_id[j]].item()))

        average_sync_score = np.mean(total_sync_scores)
        output_metrics['DeSync'] = average_sync_score

    if is_paired and paired_laion_clap_text_features is not None:
        # compute clap score
        clap_score = torch.cosine_similarity(paired_laion_clap_text_features,
                                             paired_laion_clap_audio_features,
                                             dim=-1).mean()
        output_metrics['LAION-CLAP-Score'] = clap_score.item()

        clap_score = torch.cosine_similarity(paired_ms_clap_text_features,
                                             paired_ms_clap_audio_features,
                                             dim=-1).mean()
        output_metrics['MS-CLAP-Score'] = clap_score.item()

    return output_metrics
