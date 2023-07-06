import base64
import gzip
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # x = x[:, :, :1000]
        # print(x.shape)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        all_x = []
        for block in self.blocks:
            x = block(x)
            all_x.append(torch.nn.functional.avg_pool2d(x, kernel_size=(20, 1), stride=(20, 1))[0])
        x = self.ln_post(x)
        all_x = torch.stack(all_x, dim=0) # [num_layer, pooled_time, rep_dim], e.g., [32, 75, 1280]
        return x, all_x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, at_low_compute=False):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

        if at_low_compute == False:
            self.at_model = ATModel(n_layer=self.dims.n_audio_layer, rep_dim=self.dims.n_audio_state, mode='tl_tr_1_8')
        else:
            self.at_model = ATModel(n_layer=self.dims.n_audio_layer, rep_dim=self.dims.n_audio_state, mode='tl_down_tr_512_1_8')

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function


# model related to audio tagging
class ATModel(nn.Module):
    def __init__(self, label_dim=527, n_layer=32, rep_dim=1280, mode='tl_down_tr_512_1_8'):
        super().__init__()
        self.mode = mode
        self.n_layer = n_layer
        self.rep_dim = rep_dim
        self.label_dim = label_dim

        # Time-and-Layer-Wise Transformer (TL-TR model)
        # tl_tr_1_8 = whisper original intermediate representation dim, 1-att-head time-transformer, 8-att-head layer transformer
        if 'tl_tr' in mode:
            self.num_tatt_head = int(mode.split('_')[-2])
            self.num_latt_head = int(mode.split('_')[-1])
            self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_tatt_head)
            self.layer_tr = ResidualAttentionBlock(self.rep_dim, self.num_latt_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))

        # Time-and-Layer-Wise Transformer w/ Low-Dim Projection (TL-TR-512 model)
        # lw_down_tr_512_1_8 = 512-dim rep, 1-att-head time-transformer, 8-att-head layer transformer
        if 'tl_down_tr' in mode:
            self.inter_rep_dim = int(mode.split('_')[-3])
            self.num_tatt_head = int(mode.split('_')[-2])
            self.num_latt_head = int(mode.split('_')[-1])

            self.down_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.inter_rep_dim))
            self.time_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_tatt_head)
            self.layer_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_latt_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.inter_rep_dim), nn.Linear(self.inter_rep_dim, self.label_dim))

    def forward(self, audio_rep, time_resolution=10):
        # time resolution in seconds
        # input audio_rep in shape (#layer, #time steps, rep_dim), e.g., (32, 75, 1280) # for 30 seconds, 75 = 1500 / 20 (downsampling)
        num_layer, audio_len, rep_dim = audio_rep.shape[0], audio_rep.shape[1], audio_rep.shape[2]
        decision_window = int(time_resolution * 2.5) # *100 (second-to-frame, 1 second=100frames) / 2 (whisper 2x downsample) / 20 (our 20x downsample)
        num_segment = math.ceil(audio_len / decision_window)
        target_len = num_segment * decision_window

        # padding
        padding_rows = target_len - audio_len
        if padding_rows != 0:
            audio_rep = torch.nn.functional.pad(audio_rep, (0, 0, 0, padding_rows, 0, 0), mode='constant')

        # reshape to batches for inference
        audio_rep = audio_rep.reshape([num_layer, num_segment, decision_window, rep_dim]) # [32, 3, 25, 1280]
        audio_rep = torch.permute(audio_rep, [1, 0, 2, 3]) # [3, 32, 25, 1280]
        audio_rep = audio_rep.reshape([num_segment*num_layer, decision_window, rep_dim]) # [96, 25, 1280]

        # if lower intermediate representation dimension
        if 'tl_down_tr' in self.mode:
            audio_rep = self.down_layer(audio_rep.float())  # [96, 25, 512]
        audio_rep = self.time_tr(audio_rep)  # [96, 25, 1280]
        audio_rep = torch.mean(audio_rep, dim=1)  # [96, 1280]
        audio_rep = audio_rep.reshape(num_segment, num_layer, audio_rep.shape[-1])  # [3, 32, 1280]
        audio_rep = self.layer_tr(audio_rep)  # [3, 32, 1280]
        audio_rep = torch.mean(audio_rep, dim=1)  # [3, 1280]

        pred = self.mlp_layer(audio_rep.float())  # [3, 527], 3 segments, each segment with a prediction over 527 audioset classes

        return pred