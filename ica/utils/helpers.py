from nltk.tokenize import word_tokenize
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5 import T5Model
from .Encoding import Encoding


def clean_words(sentence: str) -> [str]:
    tokens = word_tokenize(sentence)
    words = [word.lower() for word in tokens if word.isalpha()]
    return words


def clean_sentence(sentence: str) -> str:
    words = clean_words(sentence)
    sentence = f'{" ".join(words).capitalize()}.'
    return sentence


def annotate_sentence(sentence: str) -> str:
    return f'paraphrase: {sentence}'


# seq2tok
def encode_sentence(tokenizer: T5Tokenizer, sentence: str) -> BatchEncoding:
    encoded_sentence = tokenizer.batch_encode_plus(
        [sentence],
        max_length=512,
        padding='max_length',
        return_tensors="pt",
        truncation=True
    )
    return encoded_sentence

# tok2tok
def generate_sequence_of_tokens(model: T5Model, encoded: Encoding):
    return model.generate(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        do_sample=True,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1
    )[0]

# tok2seq
def decode_sequence_of_tokens(tokenizer: T5Tokenizer, sequence_of_tokens):
    return tokenizer.decode(
        sequence_of_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )