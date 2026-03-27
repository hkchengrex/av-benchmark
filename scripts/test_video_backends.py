import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from av_bench.data.video_dataset import (
    _IMAGEBIND_FPS,
    _SYNC_FPS,
    StreamingMediaDecoder,
    VideoDataset,
)

def _save_video_frames(video: torch.Tensor, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if video.ndim != 4:
        raise ValueError(f'Expected 4D video tensor, got shape={tuple(video.shape)}')
    video = video.detach().cpu()
    for i in range(video.shape[0]):
        frame = video[i].to(torch.float32)
        if frame.max() > 1.0 or frame.min() < 0.0:
            frame = frame / 255.0
        frame = frame.clamp(0.0, 1.0)
        save_image(frame, out_dir / f'{prefix}_{i:04d}.png')


def _report_pixel_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    if lhs.shape != rhs.shape:
        print(f'{name}.shape_mismatch: lhs={tuple(lhs.shape)} rhs={tuple(rhs.shape)}')
        return
    lhs_i = lhs.to(torch.int16)
    rhs_i = rhs.to(torch.int16)
    diff = (lhs_i - rhs_i).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.to(torch.float32).mean().item()
    exact_ratio = (diff == 0).to(torch.float32).mean().item()
    print(f'{name}.pixel_diff.max_abs={max_abs_diff}')
    print(f'{name}.pixel_diff.mean_abs={mean_abs_diff:.6f}')
    print(f'{name}.pixel_diff.exact_ratio={exact_ratio:.6f}')


def main():
    parser = argparse.ArgumentParser(description='Compare torio and pyav frame extraction outputs.')
    parser.add_argument('video_path', type=Path, help='Path to a video file.')
    parser.add_argument('--backend', choices=['pyav', 'torio', 'both'], default='both')
    parser.add_argument('--duration-sec', type=float, default=8.0)
    parser.add_argument('--out-dir', type=Path, default=Path('backend_decode_test'))
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f'Video file does not exist: {args.video_path}')

    dataset = VideoDataset([args.video_path], duration_sec=args.duration_sec)
    run_out_dir = args.out_dir / args.video_path.stem
    print(f'video={args.video_path}')
    print(f'duration_sec={args.duration_sec}')
    torio_ib = None
    torio_sync = None
    pyav_ib = None
    pyav_sync = None

    if args.backend in ('torio', 'both'):
        if StreamingMediaDecoder is None:
            print('torio: unavailable (skipped)')
        else:
            torio_ib, torio_sync = dataset._sample_with_torio(args.video_path)
            torio_ib_dir = run_out_dir / 'torio_ib'
            torio_sync_dir = run_out_dir / 'torio_sync'
            _save_video_frames(torio_ib, torio_ib_dir, 'frame')
            _save_video_frames(torio_sync, torio_sync_dir, 'frame')
            print(f'torio_ib.shape={tuple(torio_ib.shape)}')
            print(f'torio_sync.shape={tuple(torio_sync.shape)}')
            print(f'torio_ib.saved_dir={torio_ib_dir}')
            print(f'torio_sync.saved_dir={torio_sync_dir}')

    if args.backend in ('pyav', 'both'):
        pyav_ib = dataset._sample_with_pyav(args.video_path, _IMAGEBIND_FPS, dataset.ib_expected_length)
        pyav_sync = dataset._sample_with_pyav(args.video_path, _SYNC_FPS, dataset.sync_expected_length)
        pyav_ib_dir = run_out_dir / 'pyav_ib'
        pyav_sync_dir = run_out_dir / 'pyav_sync'
        _save_video_frames(pyav_ib, pyav_ib_dir, 'frame')
        _save_video_frames(pyav_sync, pyav_sync_dir, 'frame')
        print(f'pyav_ib.shape={tuple(pyav_ib.shape)}')
        print(f'pyav_sync.shape={tuple(pyav_sync.shape)}')
        print(f'pyav_ib.saved_dir={pyav_ib_dir}')
        print(f'pyav_sync.saved_dir={pyav_sync_dir}')

    if args.backend == 'both' and torio_ib is not None and pyav_ib is not None:
        _report_pixel_diff('ib', torio_ib, pyav_ib)
        _report_pixel_diff('sync', torio_sync, pyav_sync)


if __name__ == '__main__':
    main()
