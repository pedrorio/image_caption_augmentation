import argparse
import os
import torch
import pytorch_lightning as pl
from common.index import set_seed
from nltk import word_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer

from augmentator.DatasetAugmentator import DatasetAugmentator
from paraphraseator.UnifiedTextToTextTransformer.LoggerCallback import LoggerCallback
from paraphraseator.UnifiedTextToTextTransformer.T5FineTuner import T5FineTuner

set_seed(42)


# TODO: Before training, check if the model is already trained


class DatasetParaphraseator(DatasetAugmentator):
    """Paraphraseates the list of datasets."""

    Augmentation: DatasetAugmentator.Augmentations = "paraphraseated"

    def __init__(self, name: DatasetAugmentator.DatasetNames, augmentation_type=Augmentation,
                 recursive_levels: int = 1):
        super().__init__(name, augmentation_type, recursive_levels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').to(self.device)
        # self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser_finetuned')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.recursive_levels = recursive_levels

    @staticmethod
    def train() -> None:
        args_dict = dict(
            data_dir="", output_dir="", model_name_or_path='ramsrigouthamg/t5_paraphraser',
            tokenizer_name_or_path='t5-base', max_seq_length=512, learning_rate=3e-4, weight_decay=0.0,
            adam_epsilon=1e-8, warmup_steps=0, train_batch_size=6, eval_batch_size=6, num_train_epochs=2,
            gradient_accumulation_steps=16, n_gpu=0, early_stop_callback=False, fp_16=False, opt_level='O1',
            max_grad_norm=1.0, seed=42,
        )

        if not os.path.exists('data/finetuned'):
            os.makedirs('data/finetuned')

        args_dict.update({'data_dir': 'data/raw', 'output_dir': 'data/finetuned', 'max_seq_length': 256})
        args = argparse.Namespace(**args_dict)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir, prefix="checkpoint",
                                                           monitor="val_loss", mode="min", save_top_k=5)
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps, gpus=args.n_gpu, max_epochs=args.num_train_epochs,
            early_stop_callback=False, precision=16 if args.fp_16 else 32, amp_level=args.opt_level,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggerCallback()],
        )
        model = T5FineTuner(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
        model.model.save_pretrained('data/finetuned')

    def generate_augmentations(self, sentence: str, img_id: int) -> None:
        print(f'[{self.name}] {sentence}')
        beam_outputs = self.generate_beam_outputs(sentence)
        paraphrases = [*map(self.decode_beam_output, beam_outputs)]
        paraphrases = [*filter(
            lambda paraphrase: paraphrase not in self.sentences[img_id] and paraphrase not in self.new_sentences[
                img_id], paraphrases)]
        [*map(self.new_sentences[img_id].add, paraphrases)]
        return self.new_sentences[img_id]

    def generate_beam_outputs(self, sentence):
        max_len = min(256, int(len(word_tokenize(sentence)) * 1.5))
        encoding = self.tokenizer.encode_plus("paraphrase: " + sentence + " </s>", pad_to_max_length=True,
                                              return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(
            self.device)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_masks, do_sample=True,
                                   max_length=max_len, top_k=120, top_p=0.98, early_stopping=True,
                                   num_return_sequences=10)

    def decode_beam_output(self, beam_output):
        return self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
