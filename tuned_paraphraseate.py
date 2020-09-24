from paraphraseator.DatasetParaphraseator import DatasetParaphraseator

for dataset_name in DatasetParaphraseator.DATASET_NAMES:
    DatasetParaphraseator(dataset_name).process()
