from ica.paraphraseator.Paraphraseator import Paraphraseator

for dataset_name in Paraphraseator.DATASET_NAMES:
    Paraphraseator(dataset_name).call()