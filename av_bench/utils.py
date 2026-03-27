from pathlib import Path
from typing import Optional, Union
from collections import defaultdict
from urllib.request import urlretrieve

import torch
from tqdm import tqdm


CKPT2URL = {
    'music_speech_audioset_epoch_15_esc_89.98.pt': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt',
    'synchformer_state_dict.pth': 'https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth',
    'Cnn14_16k_mAP=0.438.pth': 'https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth',
    'Cnn14_mAP=0.431.pth': 'https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth'
}


def resolve_ckpt(ckpt: Union[str, Path]) -> Path:
    ckpt_path = Path(ckpt)

    if not ckpt_path.exists():
        if ckpt_path.name not in CKPT2URL:
            raise ValueError(f"Unknown checkpoint name: {ckpt_path.name}")
        url = CKPT2URL[ckpt_path.name]
        print(f"Downloading checkpoint from {url}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        download(url, ckpt_path)

    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    if ckpt_path.is_file():
        return ckpt_path
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def download(url: str, path: Path) -> None:
    filename = url.split("/")[-1]
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=f"Downloading {filename}",
    ) as t:
        urlretrieve(url, path, reporthook=t.update_to, data=None)


class TqdmUpTo(tqdm):
    """
    Adapted from: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5

    Alternative Class-based version of tqdm.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def clean_sample_name(sample_name: str) -> str:
    # implement your own cleaning function here to map the sample name to the sample name
    if len(sample_name) == len('000000014_zxpo56cpUBU_000007-0'):
        # extract the zxpo56cpUBU portion
        example = '000000014_zxpo56cpUBU_000007-0'
        example_target = 'zxpo56cpUBU_000007'
        start = example.find(example_target)
        end = start + len(example_target)

        vid_name = sample_name[start:end]
    elif len(sample_name) == len('zxpo56cpUBU_000007-0'):
        # extract the zxpo56cpUBU portion
        example = 'zxpo56cpUBU_000007-0'
        example_target = 'zxpo56cpUBU_000007'
        start = example.find(example_target)
        end = start + len(example_target)

        vid_name = sample_name[start:end]

    elif len(sample_name) == len('Y---g-f_I2yQ_000001_0'):
        # extract the Y---g-f_I2yQ portion
        example = 'Y---g-f_I2yQ_000001_0'
        example_target = '---g-f_I2yQ_000001'
        start = example.find(example_target)
        end = start + len(example_target)

        vid_name = sample_name[start:end]

    elif len(sample_name) == len('zxpo56cpUBU_000007'):
        vid_name = sample_name

    else:
        return sample_name

    return vid_name


def unroll_paired_dict_with_key(gt_d: dict,
                                d: dict,
                                key: str = 'logits',
                                *,
                                num_samples: Optional[int] = 10
                                ) -> tuple[list[torch.Tensor], torch.Tensor]:

    gt_features = {}
    paired_features = defaultdict(list)

    for sample_name, features in gt_d.items():
        gt_features[sample_name] = features[key]

    for sample_name, features in d.items():
        sample_name = sample_name
        paired_features[sample_name].append(features[key])

    # find the number of samples
    for sample_name, features in paired_features.items():
        if num_samples is None:
            num_samples = len(features)
        else:
            assert num_samples <= len(features)

    # combine the two dictionaries
    gt_feat_list = []
    paired_feat_list = [[] for _ in range(num_samples)]
    for sample_name, features in paired_features.items():
        if sample_name not in gt_features:
            print(f'Sample {sample_name} not found in ground truth.')
            continue
        gt_feat_list.append(gt_features[sample_name])
        for i in range(num_samples):
            paired_feat_list[i].append(features[i])

    gt_feat_list = torch.stack(gt_feat_list, dim=0)
    paired_feat_list = [torch.stack(feat_list, dim=0) for feat_list in paired_feat_list]

    return paired_feat_list, gt_feat_list


def unroll_paired_dict(gt_dict: dict,
                       pred_dict: dict,
                       cat: bool = False,
                       clean_sample_names: bool = False) -> tuple[torch.Tensor, torch.Tensor, list]:
    if clean_sample_names:
        pred_keys_to_sample = {k: clean_sample_name(k) for k in pred_dict.keys()}
    else:
        pred_keys_to_sample = {k: k for k in pred_dict.keys()}
    unpaired_samples = set(gt_dict.keys()) ^ set(pred_keys_to_sample.values())

    gt_out_list = []
    pred_out_list = []
    for key, sample_name in pred_keys_to_sample.items():
        if sample_name in unpaired_samples:
            print(f'Sample {sample_name} not found in ground truth.')
            continue
        gt_out_list.append(gt_dict[sample_name])
        pred_out_list.append(pred_dict[key])

    if cat:
        return torch.cat(gt_out_list, dim=0), torch.cat(pred_out_list,
                                                        dim=0), list(unpaired_samples)
    else:
        return torch.stack(gt_out_list, dim=0), torch.stack(pred_out_list,
                                                            dim=0), list(unpaired_samples)


def unroll_dict_all_keys(d: dict) -> dict[str, torch.Tensor]:
    out_dict = defaultdict(list)
    for k, v in d.items():
        for k2, v2 in v.items():
            out_dict[k2].append(v2)

    for k, v in out_dict.items():
        out_dict[k] = torch.stack(v, dim=0)

    return out_dict


def unroll_dict(d: dict, cat: bool = False) -> torch.Tensor:
    out_list = []
    for k, v in d.items():
        out_list.append(v)

    if cat:
        return torch.cat(out_list, dim=0)
    else:
        return torch.stack(out_list, dim=0)
