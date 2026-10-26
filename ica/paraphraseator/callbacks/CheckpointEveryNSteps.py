from os import listdir, remove
from os.path import isfile, join

from pytorch_lightning import Callback, Trainer


class CheckpointEveryNSteps(Callback):
    """Save a checkpoint every N steps and keep only the newest parsed file."""

    def __init__(self, save_step_frequency=1000, checkpoints_dir=None):
        self.save_step_frequency = max(1, int(save_step_frequency))
        self.checkpoints_dir = checkpoints_dir

    def _parsed_step(self, filename: str):
        stem = filename.split(".")[0]
        try:
            items = {item.split("=")[0]: item.split("=")[1] for item in stem.split("_")}
            return int(f'{items["epoch"]}{items["step"]}')
        except (KeyError, ValueError):
            return None

    def keep_newest_checkpoint(self):
        files = [file for file in listdir(self.checkpoints_dir) if isfile(join(self.checkpoints_dir, file))]
        scored = [(file, self._parsed_step(file)) for file in files]
        scored = [(file, step) for file, step in scored if step is not None]
        if not scored:
            return
        max_epoch_step = max(step for _, step in scored)
        for file, step in scored:
            if step < max_epoch_step:
                remove(join(self.checkpoints_dir, file))

    def on_train_batch_end(self, trainer: Trainer, _pl_module, _outputs, _batch, _batch_idx):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step > 0 and global_step % self.save_step_frequency == 0:
            self.keep_newest_checkpoint()
            file_path = (
                f"{self.checkpoints_dir}/every={self.save_step_frequency}"
                f"_epoch={epoch}_step={global_step}.ckpt"
            )
            trainer.save_checkpoint(file_path)
