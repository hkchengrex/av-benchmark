# Benchmarking for Audio-Text and Audio-Visual Generation

## Overview

This repository supports the evaluations of:

- Fréchet Distances (FD)

    - FD_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
    - FD_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), also referred to as FID/FD sometimes
    - FD_VGG, with [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md), also referred to as FAD

- Inception Scores (IS)

    - IS_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
    - IS_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), sometimes simply called IS.

- Mean KL Distances (MKL)

    - KL_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
    - KL_PANNs, with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), also referred to as KL, KLD, or MKL

- CLAP Scores

    - LAION_CLAP, cosine similarity between text and audio embeddings computed by [LAION-CLAP](https://github.com/LAION-AI/CLAP) with the `music_speech_audioset_epoch_15_esc_89.98.pt` model, following [GenAU](https://snap-research.github.io/GenAU/)
    - MS_CLAP, cosine similarity between text and audio embeddings computed by [MS-CLAP](https://github.com/microsoft/CLAP)

- ImageBind Score
    
    Cosine similarity between video and audio embeddings computed by [ImageBind](https://github.com/facebookresearch/ImageBind), sometimes scaled by 100


- DeSync Score

    Average misalignment (in seconds) predicted by [Synchformer](https://github.com/v-iashin/Synchformer) with the `24-01-04T16-39-21` model trained on AudioSet. We average the results from the first 4.8 seconds and last 4.8 seconds of each video-audio pair.

## Installation

This repository has been tested on Ubuntu, and requires **Python 3.9+** with **PyTorch 2.5.1+**. Follow the steps below to set up the environment:

### 1. Recommended Setup

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

### 2. Install PyTorch

Before proceeding, install PyTorch with the appropriate CUDA version from the [official PyTorch website](https://pytorch.org/).

### 3. Clone and Install the Repository

```bash
git clone https://github.com/hkchengrex/av-benchmark.git
cd av-benchmark
pip install -e .
```

### 4. Download Pretrained Models

Download [music_speech_audioset_epoch_15_esc_89.98.pt](https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt) and [Synchformer](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth) and put them in `weights`.

(Alternatively, execute the following when you are in the root directory of this repository)

```bash
mkdir weights
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt -O weights/music_speech_audioset_epoch_15_esc_89.98.pt
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth -O weights/synchformer_state_dict.pth
```

### 5. Optional: For Video Evaluation

If you plan to evaluate on videos, you will also need `ffmpeg`. Note that torchaudio imposes a maximum version limit (`ffmpeg<7`). You can install it as follows:

```bash
conda install -c conda-forge 'ffmpeg<7'
```

## Usage

### Overview

Evaluation is a two-stage process:

1. **Extraction**: extract video/text/audio features for ground-truth and audio features for the predicted audios. The extracted features are saved in `gt_cache` and `pred_cache` respectively.
2. **Evaluation**: compute the desired metrics using the extracted features.

By default, if `gt_cache` or `pred_cache` are not specified, we will use `gt_audio/cache` and `pred_audio/cache`.
`gt_audio` and `pred_audio` should point to a directory containing audio files in either `.wav` or `.flac` formats.

### Extraction

#### 1. **Video feature extraction (optional).**
For video-to-audio applications, visual features are extracted from input videos. This is also applicable for generated videos in audio-to-video or audio-visual joint generation tasks.

**Input requirements:**

- Videos in .mp4 format (any FPS or resolution).
- Video names must match the corresponding audio file names (excluding extensions).

Run the following to extract visual features using `Synchformer` and `ImageBind`:

```bash
python extract_video.py --gt_cache <output cache directory> --video_path <directory containing videos> --gt_batch_size <batch size> --audio_length=<length of video in seconds>
```

Some of the precomputed caches for VGGSound/AudioCaps can be found here: https://huggingface.co/datasets/hkchengrex/MMAudio-precomputed-results

#### 2. **Text feature extraction (optional).**
For text-to-audio applications, text features are extracted from input text data.

**Input requirements:**

- A CSV file with at least two columns with a header row:
    - `name`: Matches the corresponding audio file name (excluding extensions).
    - `caption`: The text associated with the audio.

Run the following to extract text features using `LAION-CLAP` and `MS-CLAP`:

```bash
python extract_text.py --text_csv <path to the csv> --output_cache_path <output cache directory>
```

#### 3. **Audio feature extraction.**

Audio features are automatically extracted during the evaluation stage.

**Manual extraction:**
You can force feature extraction by specifying:
 - `--recompute_gt_cache` for ground-truth audio features.
 - `--recompute_pred_cache` for predicted audio features.

This is useful if the extraction is interrupted or the cache is corrupted.

### Evaluation

```bash
python evaluate.py  --gt_audio <gt audio path> --gt_cache <gt cache path> --pred_audio <pred audio path> --pred_cache <pred cache path> --audio_length=<length of audio wanted in seconds> 
```

You can specify `--skip_clap` or `--skip_video_related` to speed up evaluation if you don't need those metrics.

## Supporting Libraries

To address issues with deprecated code in some underlying libraries, we have forked and modified several of them. These forks are included as dependencies to ensure compatibility down the road.

- LAION-CLAP: https://github.com/hkchengrex/CLAP
- MS-CLAP: https://github.com/hkchengrex/MS-CLAP
- PaSST: https://github.com/hkchengrex/passt_hear21
- ImageBind: https://github.com/hkchengrex/ImageBind


## Citation

This repository is part of the accompanying code for MMAudio. To cite this repository, please use the following BibTeX entry:

```bibtex
@inproceedings{cheng2024taming,
  title={MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis},
  author={Cheng, Ho Kei and Ishii, Masato and Hayakawa, Akio and Shibuya, Takashi and Schwing, Alexander and Mitsufuji, Yuki},
  booktitle={CVPR},
  year={2025}
}
```


## References

Many thanks to
- [PaSST](https://github.com/kkoutini/PaSST)
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
- [passt_hear21](https://github.com/kkoutini/passt_hear21)
- [torchvggish](https://github.com/harritaylor/torchvggish)
- [audioldm_eval](https://github.com/haoheliu/audioldm_eval) -- on which this repository is based on
- [LAION-CLAP](https://github.com/LAION-AI/CLAP)
- [MS-CLAP](https://github.com/microsoft/CLAP)
- [ImageBind](https://github.com/facebookresearch/ImageBind)
- [Synchformer](https://github.com/v-iashin/Synchformer)
