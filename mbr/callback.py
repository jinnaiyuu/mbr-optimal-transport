from typing import Dict, Any, Optional, Union
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)


class SacredLoggingCallback(TrainerCallback):
    """
    A callback for logging training metrics to Sacred experiment tracking.

    This callback integrates with the Sacred experiment tracking framework to log
    metrics during the training process of a Hugging Face Transformers model.
    """

    def __init__(self, _run: Any) -> None:
        """
        Initialize the Sacred logging callback.

        Args:
            _run: The Sacred experiment run object used for logging metrics.
        """
        self._run = _run

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """
        Called at the beginning of training.

        Args:
            args: The training arguments.
            state: The current state of the trainer.
            control: The trainer control object.
            **kwargs: Additional arguments.
        """
        print("Training begins...")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """
        Called at the end of an epoch.

        Args:
            args: The training arguments.
            state: The current state of the trainer.
            control: The trainer control object.
            **kwargs: Additional arguments.
        """
        print(f"Epoch {state.epoch} ended.")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when the trainer logs metrics.

        This method logs scalar values to Sacred and prints a warning for non-scalar values.

        Args:
            args: The training arguments.
            state: The current state of the trainer.
            control: The trainer control object.
            logs: Dictionary of metrics to log.
            **kwargs: Additional arguments.
        """
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self._run.log_scalar(key, value, step=state.global_step)
                else:
                    print("Trying to log a non-scalar value:", key, value)
