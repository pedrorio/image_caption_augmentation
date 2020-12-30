import os
import logging
from pytorch_lightning import Callback, seed_everything

seed_everything(42)


class LoggerCallback(Callback):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_validation_end(self, trainer, module):
        self.logger.info("***** Validation results *****")
        if self.is_logger(trainer):
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, module):
        self.logger.info("***** Test results *****")
        if self.is_logger(trainer):

            metrics = trainer.callback_metrics
            output_test_results_file = os.path.join(
                module.output_dir,
                "test_results.txt"
            )
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        self.logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

    def is_logger(self, trainer):
        return trainer.global_rank <= 0

