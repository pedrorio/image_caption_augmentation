from pytorch_lightning.callbacks import ModelCheckpoint


def checkpoint_callback(checkpoints_dir: str):
    return ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=5
    )
