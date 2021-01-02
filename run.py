from ica.paraphraseator.T5 import T5

t5 = T5(
        gpus=1,
        num_workers=4,
        batch_size=5,
        data_dir="/content/drive/MyDrive/ica/data/raw",
        logs_dir="/content/drive/MyDrive/ica/data/logs",
        cache_dir="/content/drive/MyDrive/ica/data/cache",
        checkpoints_dir="/content/drive/MyDrive/ica/data/checkpoints"
    )
t5.train_model()
