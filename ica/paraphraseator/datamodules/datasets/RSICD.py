import os
from .ImageCaptionsDataset import ImageCaptionsDataset


def RSICD(data_dir: str, transform=None):
    file_name = 'dataset_rsicd_modified.json'
    file_path = os.path.join(data_dir, file_name)
    # f'{data_path}/{file_name}'
    return ImageCaptionsDataset(file_path=file_path, transform=transform)