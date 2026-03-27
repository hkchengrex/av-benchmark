import logging
from pathlib import Path

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2

try:
    from torio.io import StreamingMediaDecoder
except ImportError:
    StreamingMediaDecoder = None

try:
    import av as pyav
except ImportError:
    pyav = None

if StreamingMediaDecoder is not None:
    _VIDEO_BACKEND = 'torio'
elif pyav is not None:
    _VIDEO_BACKEND = 'pyav'
else:
    raise ImportError(
        'torio is unavailable and pyav could not be imported. '
        'Please install pyav (e.g., `pip install av`)'
    ) from None
from av_bench.data.ib_data import SpatialCrop

log = logging.getLogger()
if _VIDEO_BACKEND == 'pyav':
    log.warning('torio is unavailable; falling back to pyav video decoding.')

# https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py
# https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/functional.html
_IMAGEBIND_SIZE = 224
_IMAGEBIND_FPS = 0.5

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class VideoDataset(Dataset):

    def __init__(
        self,
        video_paths: list[Path],
        *,
        duration_sec: float = 8.0,
    ):
        self.video_paths = video_paths

        self.duration_sec = duration_sec

        self.ib_expected_length = int(_IMAGEBIND_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.ib_transform = v2.Compose([
            v2.Resize(_IMAGEBIND_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.crop = SpatialCrop(_IMAGEBIND_SIZE, 3)

    def _sample_with_torio(self, video_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_IMAGEBIND_FPS * self.duration_sec),
            frame_rate=_IMAGEBIND_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        ib_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        return ib_chunk, sync_chunk

    def _sample_with_pyav(
        self,
        video_path: Path,
        frame_rate: float,
        expected_length: int,
    ) -> torch.Tensor:
        if pyav is None:
            raise ImportError('pyav is not installed. Install it with `pip install av`.')
        container = pyav.open(str(video_path))
        video_stream = container.streams.video[0]
        try:
            graph = pyav.filter.Graph()
            source = graph.add_buffer(template=video_stream)
            fps_filter = graph.add('fps', f'fps={frame_rate}:round=near:start_time=0')
            fmt_filter = graph.add('format', 'pix_fmts=rgb24')
            sink = graph.add('buffersink')
            source.link_to(fps_filter)
            fps_filter.link_to(fmt_filter)
            fmt_filter.link_to(sink)
            graph.configure()
        except Exception as e:
            raise RuntimeError(f'Unable to build PyAV filter graph for {video_path}: {e}') from e

        frames = []
        try:
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                _ = frame_idx
                source.push(frame)
                while (
                    len(frames) < expected_length
                ):
                    try:
                        out_frame = sink.pull()
                    except Exception as pull_exc:
                        if pull_exc.__class__.__name__ in {'BlockingIOError', 'EOFError', 'FFmpegError'}:
                            break
                        raise
                    frames.append(torch.from_numpy(out_frame.to_ndarray(format='rgb24')))
                if len(frames) >= expected_length:
                    break

            source.push(None)
            while len(frames) < expected_length:
                try:
                    out_frame = sink.pull()
                except Exception as pull_exc:
                    if pull_exc.__class__.__name__ in {'BlockingIOError', 'EOFError', 'FFmpegError'}:
                        break
                    raise
                frames.append(torch.from_numpy(out_frame.to_ndarray(format='rgb24')))

            if len(frames) < expected_length:
                raise RuntimeError(
                    f'Video too short {video_path}, expected at least {expected_length} frames at {frame_rate} fps'
                )
            # Match torio output layout: (T, C, H, W).
            return torch.stack(frames[:expected_length]).permute(0, 3, 1, 2)
        finally:
            container.close()

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_path = self.video_paths[idx]

        if _VIDEO_BACKEND == 'torio':
            ib_chunk, sync_chunk = self._sample_with_torio(video_path)
        elif _VIDEO_BACKEND == 'pyav':
            ib_chunk = self._sample_with_pyav(video_path, _IMAGEBIND_FPS, self.ib_expected_length)
            sync_chunk = self._sample_with_pyav(video_path, _SYNC_FPS, self.sync_expected_length)
        if ib_chunk is None:
            raise RuntimeError(f'IB video returned None {video_path}')
        if ib_chunk.shape[0] < self.ib_expected_length:
            raise RuntimeError(
                f'IB video too short {video_path}, expected {self.ib_expected_length}, got {ib_chunk.shape[0]}'
            )
        if ib_chunk.ndim != 4 or ib_chunk.shape[1] != 3:
            raise RuntimeError(
                f'IB video has unexpected shape {tuple(ib_chunk.shape)}; expected (T,C,H,W)'
            )

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_path}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f'Sync video too short {video_path}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}'
            )
        if sync_chunk.ndim != 4 or sync_chunk.shape[1] != 3:
            raise RuntimeError(
                f'Sync video has unexpected shape {tuple(sync_chunk.shape)}; expected (T,C,H,W)'
            )
        
        # truncate the video
        ib_chunk = ib_chunk[:self.ib_expected_length]
        if ib_chunk.shape[0] != self.ib_expected_length:
            raise RuntimeError(f'IB video wrong length {video_path}, '
                               f'expected {self.ib_expected_length}, '
                               f'got {ib_chunk.shape[0]}')
        ib_chunk = self.ib_transform(ib_chunk)

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_path}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        ib_chunk = self.crop([ib_chunk])
        ib_chunk = torch.stack(ib_chunk)

        data = {
            'name': video_path.stem,
            'ib_video': ib_chunk,
            'sync_video': sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.video_paths[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.video_paths)
