from typing import List

import nltk
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from .Encoding import Encoding


def ensure_nltk() -> None:
    """Download the tokenizers used by caption cleaning. Missing data is the only error we handle."""
    for resource, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_words(sentence: str) -> List[str]:
    ensure_nltk()
    tokens = nltk.word_tokenize(sentence)
    return [word.lower() for word in tokens if word.isalpha()]


def clean_sentence(sentence: str) -> str:
    words = clean_words(sentence)
    return f'{" ".join(words).capitalize()}.'


def annotate_sentence(sentence: str) -> str:
    return f"paraphrase: {sentence}"


def encode_sentence(tokenizer: T5Tokenizer, sentence: str, max_length: int = 512) -> dict:
    encoded_sentence = tokenizer(
        sentence,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )
    return {
        "input_ids": encoded_sentence["input_ids"].squeeze(0),
        "attention_mask": encoded_sentence["attention_mask"].squeeze(0),
    }


def generate_kwargs_from_settings(
    do_sample: bool = True,
    top_k: int = 120,
    top_p: float = 0.98,
    early_stopping: bool = True,
    num_return_sequences: int = 1,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
) -> dict:
    settings = {
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "early_stopping": early_stopping,
        "num_return_sequences": num_return_sequences,
        "max_new_tokens": max_new_tokens,
    }
    if do_sample:
        settings["temperature"] = temperature
    return settings


def generate_sequence_of_tokens(model, encoded: Encoding, **generate_kwargs):
    settings = generate_kwargs_from_settings()
    settings.update(generate_kwargs)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and settings.get("max_new_tokens") is not None:
        generation_config.max_length = None
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **settings,
    )
    return sequences[0]


def decode_sequence_of_tokens(tokenizer: T5Tokenizer, sequence_of_tokens) -> str:
    return tokenizer.decode(
        sequence_of_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
