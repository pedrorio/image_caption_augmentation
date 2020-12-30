import argparse
import os
import torch
from pytorch_lightning import Trainer, callbacks, seed_everything
from ..utils import Augmentator
from .loggers import LoggerCallback

seed_everything(42)


class Paraphraseator(Augmentator):
    """Paraphraseates the list of datasets."""
    Augmentation: Augmentator.Augmentations = "paraphraseated"

    def __init__(
            self,
            name: Augmentator.DatasetNames,
            augmentation_type=Augmentation,
            recursive_levels: int = 1,
            finetuned: bool = False
    ):
        super().__init__(name, augmentation_type, recursive_levels)
        self.device = torch.device("cpu")
        self.recursive_levels = recursive_levels
        self.finetuned = finetuned

        self.data_dir = 'data/raw'
        self.output_dir = select_output_dir(self.finetuned)

        self.t5 = T5()

    @staticmethod
    def train() -> None:
        filepath = 'data/paraphrase'
        os.makedirs(filepath) if not os.path.exists(filepath) else None
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=filepath,
            prefix="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=5
        )
        datamodule_params = dict(
            # model_name_or_path='ramsrigouthamg/t5_paraphraser',
            # tokenizer_name_or_path='t5-base',
            # num_train_epochs=2,
            # warmup_steps=0,
            # gradient_accumulation_steps=16,
            max_seq_length=512,
            # train_batch_size=6,
            # eval_batch_size=6,
            # data_dir='data/raw',
            # num_workers=0,
        )
        model_params = dict(
            # learning_rate=3e-4,
            # adam_epsilon=1e-8,
            # weight_decay=0.0,
            # model_name_or_path='ramsrigouthamg/t5_paraphraser',
            # tokenizer_name_or_path='t5-base',
            # train_batch_size=6,
            # eval_batch_size=6,
            # gpus=0,
            # accumulate_grad_batches=16,
            # max_epochs=2
        )
        model = Model(argparse.Namespace(**model_params))

        train_params = dict(
            # accumulate_grad_batches=16,
            # max_epochs=2,
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggerCallback()],
            # gpus=0,
            early_stop_callback=False,
            gradient_clip_val=1.0
        )
        # trainer = Trainer(**train_params)
        #
        # trainer.fit(model, datamodule)
        # model.model.save_pretrained(filepath)

    def generate_augmentations(self, sentence: str, img_id: int) -> None:
        print(f'[{self.name}] {sentence}')
        self.new_sentences[img_id] = {
            paraphrase for paraphrase in (
                self.t5.generate(sentence)
            ) if self.filter_sentences(paraphrase, img_id)
        }

    def select_model(self, finetuned: bool):
        if finetuned:
            model = T5ForConditionalGeneration.load_from_checkpoint('data/finetuned')
            model.freeze()
        else:
            model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').to(self.device)
        return model

    def select_output_dir(self, finetuned: bool):
        if finetuned:
            return 'data/finetuned'
        return 'data/experiment'

    # @staticmethod
    # def generate_sentence_encoding(sentence: str) -> str:
    #     return f'paraphrase: {sentence} </s>'
