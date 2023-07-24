import hashlib
import io
import os
import urllib
import warnings
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from .audio import load_audio, log_mel_spectrogram, pad_or_trim
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import ModelDimensions, Whisper
from .transcribe import transcribe
from .at_post_processing import *
from .version import __version__

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}

_MODELS_AT = {
    "tiny.en": "https://www.dropbox.com/s/atq9so6w0qug5ai/tiny.en_ori.pth?dl=1",
    "tiny": "https://www.dropbox.com/s/cib4q4iz6g758l0/tiny_ori.pth?dl=1",
    "base.en": "https://www.dropbox.com/s/qtzgsbuquoz0afn/base.en_ori.pth?dl=1",
    "base": "https://www.dropbox.com/s/2odwh42u6e9ger7/base_ori.pth?dl=1",
    "small.en": "https://www.dropbox.com/s/cyx50ycl1ul7lji/small.en_ori.pth?dl=1",
    "small.en_low": "https://www.dropbox.com/s/507o66zgl8v6ddd/small.en_low.pth?dl=1",
    "small": "https://www.dropbox.com/s/jftj9s0kr4ycvr1/small_ori.pth?dl=1",
    "small_low": "https://www.dropbox.com/s/a1x0416v58f7wrf/small_low.pth?dl=1",
    "medium.en": "https://www.dropbox.com/s/bbvylvmgns8ja4p/medium.en_ori.pth?dl=1",
    "medium.en_low": "https://www.dropbox.com/s/2q5wprr8f9gti5t/medium.en_low.pth?dl=1",
    "medium": "https://www.dropbox.com/s/65aabayr7o819az/medium_ori.pth?dl=1",
    "medium_low": "https://www.dropbox.com/s/0mnfmcasram4n6o/medium_low.pth?dl=1",
    "large-v1": "https://www.dropbox.com/s/b8x2en1fdzc8nhk/large-v1_ori.pth?dl=1",
    "large-v1_low": "https://www.dropbox.com/s/5o79h70wyla8jlk/large-v1_low.pth?dl=1",
    "large-v2": "https://www.dropbox.com/s/3zxpyvdrxy22eq7/large-v2_ori.pth?dl=1",
    "large-v2_low": "https://www.dropbox.com/s/jw2rh4uylhqgn85/large-v2_low.pth?dl=1",
    "large": "https://www.dropbox.com/s/3zxpyvdrxy22eq7/large-v2_ori.pth?dl=1",
    "large_low": "https://www.dropbox.com/s/jw2rh4uylhqgn85/large-v2_low.pth?dl=1",
}

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
}


def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    parsed_url = urllib.parse.urlparse(url).path
    download_target = os.path.join(root, os.path.basename(parsed_url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        #if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
        return model_bytes if in_memory else download_target
        # else:
        #     warnings.warn(
        #         f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
        #     )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    # if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
    #     raise RuntimeError(
    #         "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
    #     )

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
    at_low_compute = False
) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    # if use low-dim proj, only applied for large, medium, and small model
    if at_low_compute == True:
        at_mdl_name = name + '_low'
    else:
        at_mdl_name = name

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        checkpoint_file_at = _download(_MODELS_AT[at_mdl_name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    with (
        io.BytesIO(checkpoint_file_at) if in_memory else open(checkpoint_file_at, "rb")
    ) as fp:
        checkpoint_at = torch.load(fp, map_location=device)
    del checkpoint_file_at

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims, at_low_compute=at_low_compute)

    combined_state_dict = {}
    combined_state_dict.update(checkpoint["model_state_dict"])
    combined_state_dict.update(checkpoint_at)

    model.load_state_dict(combined_state_dict, strict=True)

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)
