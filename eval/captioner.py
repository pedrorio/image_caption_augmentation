"""Frozen ResNet-18 features plus a one-layer GRU caption decoder.

ResNet-18 is the vision encoder because its ImageNet weights are small, the
features can be cached, and the A/B then trains only the decoder. A ViT-GPT2
captioner would be a different system from the one this CPU check can run.
"""

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from eval.records import CaptionRecord
from eval.text import tokenize_caption


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def pad(self) -> int:
        return 0

    @property
    def bos(self) -> int:
        return 1

    @property
    def eos(self) -> int:
        return 2

    @property
    def unk(self) -> int:
        return 3

    @classmethod
    def build(cls, sentences: list[str]) -> "Vocab":
        words: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            for token in tokenize_caption(sentence):
                if token not in seen:
                    seen.add(token)
                    words.append(token)
        stoi = {word: index + 4 for index, word in enumerate(words)}
        itos = {index: word for word, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, sentence: str, max_len: int) -> list[int]:
        ids = [self.bos]
        for token in tokenize_caption(sentence):
            if len(ids) >= max_len - 1:
                break
            ids.append(self.stoi.get(token, self.unk))
        ids.append(self.eos)
        return ids

    def __len__(self) -> int:
        return 4 + len(self.stoi)


class GruCaptioner(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, feat_dim: int = 512):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.feat_proj(features)).unsqueeze(0)
        embedded = self.embed(tokens)
        outputs, _hidden = self.gru(embedded, hidden)
        return self.out(outputs)


def extract_features(records: list[CaptionRecord], device: torch.device) -> np.ndarray:
    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    encoder.to(device)
    transform = weights.transforms()
    rows = []
    for record in records:
        path = record.image_path
        if path is None:
            raise FileNotFoundError(f"no image file for {record.image_id}")
        with Image.open(path) as image:
            batch = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = encoder(batch).squeeze(0).detach().cpu().numpy()
        rows.append(feature.astype(np.float32))
    return np.stack(rows)


def _pad(sequences: list[list[int]], pad: int) -> torch.Tensor:
    width = max(len(sequence) for sequence in sequences)
    out = torch.full((len(sequences), width), pad, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        out[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return out


def train_decoder(
    features: np.ndarray,
    examples: list[tuple[int, list[int]]],
    vocab: Vocab,
    embed_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> GruCaptioner:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GruCaptioner(len(vocab), embed_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad)
    model.train()
    for _epoch in range(epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[start : start + batch_size]]
            feats = torch.tensor(np.stack([features[index] for index, _ids in batch]), dtype=torch.float32, device=device)
            tokens = _pad([ids for _index, ids in batch], vocab.pad).to(device)
            if tokens.size(1) < 2:
                continue
            logits = model(feats, tokens[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def greedy_caption(
    model: GruCaptioner,
    feature: np.ndarray,
    vocab: Vocab,
    max_len: int,
    device: torch.device,
) -> str:
    model.eval()
    feats = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
    hidden = torch.tanh(model.feat_proj(feats)).unsqueeze(0)
    token = torch.tensor([[vocab.bos]], dtype=torch.long, device=device)
    pieces: list[str] = []
    for _step in range(max_len):
        outputs, hidden = model.gru(model.embed(token), hidden)
        nxt = int(model.out(outputs[:, -1]).argmax(dim=-1).item())
        if nxt == vocab.eos:
            break
        if nxt == vocab.unk:
            pieces.append("<unk>")
        elif nxt not in (vocab.pad, vocab.bos):
            word = vocab.itos.get(nxt)
            if word:
                pieces.append(word)
        token = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return " ".join(pieces)
