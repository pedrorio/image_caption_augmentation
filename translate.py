from translator.DatasetTranslator import DatasetTranslator

for dataset_name in DatasetTranslator.DATASET_NAMES:
    DatasetTranslator(dataset_name).process()
