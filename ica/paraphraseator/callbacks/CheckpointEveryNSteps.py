from pytorch_lightning import Callback, Trainer
from os import listdir, remove
from os.path import isfile, join


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

    def keep_newest_checkpoint(self):
        files = [file for file in listdir(self.checkpoints_dir) if isfile(join(self.checkpoints_dir, file))]
        files_to_delete = []
        max_epoch_step = 0

        for file in files:
            items = file.split(".")[0].split("_")
            items = {item.split("=")[0]: item.split("=")[1] for item in items}
            file_epoch_step = int(f'{items["epoch"]}{items["step"]}')

            if file_epoch_step > max_epoch_step:
                max_epoch_step = file_epoch_step

        for file in files:
            items = file.split(".")[0].split("_")
            items = {item.split("=")[0]: item.split("=")[1] for item in items}
            file_epoch_step = int(f'{items["epoch"]}{items["step"]}')

            if file_epoch_step < max_epoch_step:
                files_to_delete.append(file)

        print(max_epoch_step)
        print(files_to_delete)

        for file in files_to_delete:
            remove(join(self.checkpoints_dir, file))

    def on_batch_end(self, trainer: Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.keep_newest_checkpoint()
            file_path = f"{self.checkpoints_dir}/every={self.save_step_frequency}_epoch={epoch}_step={global_step}.ckpt"
            trainer.save_checkpoint(file_path)
