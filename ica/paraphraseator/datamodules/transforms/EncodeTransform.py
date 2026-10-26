from ica.utils.helpers import annotate_sentence, clean_sentence, encode_sentence


class EncodeTransform:
    def __init__(self, tokenizer, max_source_length: int = 64, max_target_length: int = 64):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, sample):
        x, y = sample

        clean_x = clean_sentence(x)
        annotated_x = annotate_sentence(clean_x)
        encoded_x = encode_sentence(self.tokenizer, annotated_x, max_length=self.max_source_length)

        clean_y = clean_sentence(y)
        encoded_y = encode_sentence(self.tokenizer, clean_y, max_length=self.max_target_length)

        return dict(
            x=x,
            clean_x=clean_x,
            annotated_x=annotated_x,
            encoded_x=encoded_x,
            y=y,
            clean_y=clean_y,
            encoded_y=encoded_y,
        )
