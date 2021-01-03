import os
from pytorch_lightning import Callback, Trainer


class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            save_step_frequency=1000,
            checkpoints_dir=None
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
        """
        self.save_step_frequency = save_step_frequency
        self.checkpoints_dir = checkpoints_dir

    def on_batch_end(self, trainer: Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            file_path = f"{self.checkpoints_dir}/every={self.save_step_frequency}_epoch={epoch}_step={global_step}.ckpt"
            trainer.save_checkpoint(file_path)
