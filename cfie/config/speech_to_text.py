# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from cfie.config.utils import config


@config
class SpeechToTextConfig:
    """Configuration for speech-to-text models."""

    # 输入音频将被重采样到这个采样率。
    sample_rate: float = 16_000
    """Sample rate (Hz) to resample input audio to. Most speech models expect
    16kHz audio input. The input audio will be automatically resampled to this
    rate before processing."""

    # 单个音频片段允许的最大时长；超过时可能被切块。
    max_audio_clip_s: int | None = 30
    """Maximum duration in seconds for a single audio clip without chunking.
    Audio longer than this will be split into smaller chunks if
    `allow_audio_chunking` evaluates to True, otherwise it will be rejected. 
    `None` means audio duration can be unlimited and won't be chunked."""

    # 相邻音频块之间保留的重叠秒数。
    overlap_chunk_second: int = 1
    """Overlap duration in seconds between consecutive audio chunks when
    splitting long audio. This helps maintain context across chunk boundaries
    and improves transcription quality at split points."""

    # 在静音区域找切分点时使用的搜索窗口大小。
    min_energy_split_window_size: int | None = 1600
    """Window size in samples for finding low-energy (quiet) regions to split
    audio chunks. The algorithm looks for the quietest moment within this
    window to minimize cutting through speech. Default 1600 samples ≈ 100ms
    at 16kHz. If None, no chunking will be done."""

    @property
    def allow_audio_chunking(self) -> bool:
        # 只有同时给出“静音切分窗口”和“单块最大时长”时，才允许自动切块。
        return (
            self.min_energy_split_window_size is not None
            and self.max_audio_clip_s is not None
        )
