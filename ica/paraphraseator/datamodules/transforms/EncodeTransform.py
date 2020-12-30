from nltk.tokenize import word_tokenize
from ....utils.EncodedSentence import EncodedSentence

class EncodeTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sample):
        x, y = sample

        def clean_sentence(sentence):
            tokens = word_tokenize(sentence)
            words = [word.lower() for word in tokens if word.isalpha()]
            sentence = f'{" ".join(words).capitalize()}.'
            return sentence

        x = clean_sentence(x)
        y = clean_sentence(y)

        def annotate_source(x):
            # return f'paraphrase: {x} </s>'
            return f'paraphrase: {x}'

        x = annotate_source(x)

        def annotate_target(y):
            # return f'{y} </s>'
            return f'{y}'

        y = annotate_target(y)

        def encode(sentence):
            encoded_sentence = self.tokenizer.batch_encode_plus(
                [sentence],
                max_length=512,
                padding='max_length',
                return_tensors="pt",
                truncation=True
            )
            return encoded_sentence

        x = encode(x)
        y = encode(y)

        x_inputs, x_attention = x.input_ids.squeeze(), x.attention_mask.squeeze()
        y_inputs, y_attention = y.input_ids.squeeze(), y.attention_mask.squeeze()

        sample = dict(
            x_inputs=x_inputs,
            x_attention=x_attention,
            y_inputs=y_inputs,
            y_attention=y_attention
        )
        return sample
