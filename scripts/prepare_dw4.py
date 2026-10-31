"""Download the DW4 dataset of Klein et al. (2023) from OSF and convert it to plain .npy arrays.

The original file is a pickled object array containing a torch tensor. It is loaded with a restricted
unpickler that only allows numpy/torch array reconstruction, and re-saved without pickle:
    data/dw4_data.npy  float32 [1_000_000, 4, 2]   MCMC samples (not centered)
    data/dw4_idx.npy   int64   [1_000_000]         fixed permutation used for the train/test split
"""

import codecs
import collections
import io
import os
import pickle
import urllib.request

import numpy as np
import torch

URL = "https://osf.io/download/mus7z/?view_only=28deeba0845546fb96d1b2f355db0da5"
DATA_DIR = "data"

ALLOWED = {
    ("_codecs", "encode"): codecs.encode,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.storage", "_load_from_bytes"): lambda b: torch.load(io.BytesIO(b), weights_only=True),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return ALLOWED[(module, name)]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(f"{DATA_DIR}/dw4_data.npy") and os.path.exists(f"{DATA_DIR}/dw4_idx.npy"):
        print("DW4 already prepared")
        return
    raw = f"{DATA_DIR}/dw4-dataidx.npy"
    if not os.path.exists(raw):
        print("downloading DW4 from OSF ...")
        urllib.request.urlretrieve(URL, raw)
    with open(raw, "rb") as f:
        version = np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        data, idx = SafeUnpickler(f).load()
    data = data.numpy().reshape(-1, 4, 2).astype(np.float32)
    np.save(f"{DATA_DIR}/dw4_data.npy", data)
    np.save(f"{DATA_DIR}/dw4_idx.npy", np.asarray(idx).astype(np.int64))
    print("saved", data.shape)


if __name__ == "__main__":
    main()
