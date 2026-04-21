# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from pathlib import Path

import cfie.envs as envs
from cfie.connections import global_http_connection

VLLM_S3_BUCKET_URL = "https://cfie-public-assets.s3.us-west-2.amazonaws.com"


def get_cache_dir() -> Path:
    """Get the path to the cache for storing downloaded assets."""
    path = Path(envs.VLLM_ASSETS_CACHE)
    path.mkdir(parents=True, exist_ok=True)

    return path


@lru_cache
def get_cfie_public_assets(filename: str, s3_prefix: str | None = None) -> Path:
    """
    Download an asset file from `s3://cfie-public-assets`
    and return the path to the downloaded file.
    """
    asset_directory = get_cache_dir() / "cfie_public_assets"
    asset_directory.mkdir(parents=True, exist_ok=True)

    asset_path = asset_directory / filename
    if not asset_path.exists():
        if s3_prefix is not None:
            filename = s3_prefix + "/" + filename
        global_http_connection.download_file(
            f"{VLLM_S3_BUCKET_URL}/{filename}",
            asset_path,
            timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )

    return asset_path
