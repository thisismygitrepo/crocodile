import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # Added TensorDataset for example
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter # For TensorBoard logging
import numpy as np
# import time
import os
import copy
from tqdm.auto import tqdm # For progress bars
from typing import Any, TypeVar, Generic, Callable, Optional, Union, Dict, List, Tuple

T = TypeVar('T')

# Type for callback logs
LogsType = Optional[Dict[str, Any]]


# --- Callback System ---
class Callback:
    def __init__(self):
        self.trainer: Optional['AdvancedModelTrainer[Any]'] = None # Will be set by the trainer
    def set_trainer(self, trainer: 'AdvancedModelTrainer[Any]') -> None:
        self.trainer = trainer
    def on_train_begin(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_train_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_epoch_begin(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_epoch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_batch_begin(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_batch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_validation_begin(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass
    def on_validation_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None: pass


class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', patience: int = 5, mode: str = 'min', verbose: bool = True, delta: float = 0, restore_best_weights: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
        self.best_weights = None

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        if delta < 0:
            raise ValueError("delta must be non-negative")

    def on_epoch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            print(f"Warning: Early stopping monitor '{self.monitor}' not found in logs. Available keys: {list(logs.keys())}")
            return

        if epoch is None:
            return

        if self.mode == 'min':
            score_improved = current_score < self.best_score - self.delta
        else: # mode == 'max'
            score_improved = current_score > self.best_score + self.delta

        if score_improved:
            self.best_score = current_score
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.trainer.model.state_dict())  # type: ignore[union-attr]
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} improved to {current_score:.4f}. Saving model weights.")
        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best_score:.4f}. Patience {self.wait}/{self.patience}.")
            if self.wait >= self.patience:
                self.trainer.stop_training = True  # type: ignore[union-attr]
                print(f"Epoch {epoch+1}: Early stopping triggered after {self.patience} epochs of no improvement.")
                if self.restore_best_weights and self.best_weights is not None:
                    print(f"Restoring model weights from epoch {self.best_epoch+1} with {self.monitor} = {self.best_score:.4f}")
                    self.trainer.model.load_state_dict(self.best_weights)  # type: ignore[union-attr]


class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = True, save_weights_only: bool = True, verbose: bool = True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.best_score = np.inf if mode == 'min' else -np.inf
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            print(f"Warning: ModelCheckpoint monitor '{self.monitor}' not found in logs. Available keys: {list(logs.keys())}")
            return

        if epoch is None:
            return

        if self.mode == 'min':
            improved = current_score < self.best_score
        else:
            improved = current_score > self.best_score

        filepath_epoch = self.filepath.format(epoch=epoch+1, **logs)

        if self.save_best_only:
            if improved:
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best_score:.4f} to {current_score:.4f}, saving model to {filepath_epoch}")
                self.best_score = current_score
                self._save_model(filepath_epoch)
        else:
            if self.verbose:
                print(f"Epoch {epoch+1}: saving model to {filepath_epoch}")
            self._save_model(filepath_epoch)

        # Always save last model for resuming
        last_filepath = os.path.join(os.path.dirname(self.filepath), "last_checkpoint.pth")
        self._save_model(last_filepath, is_checkpoint=True, epoch=epoch, logs=logs)


    def _save_model(self, path: str, is_checkpoint: bool = False, epoch: Optional[int] = None, logs: LogsType = None) -> None:
        if self.save_weights_only and not is_checkpoint:
            torch.save(self.trainer.model.state_dict(), path)  # type: ignore[union-attr]
        else:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.trainer.model.state_dict(),  # type: ignore[union-attr]
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),  # type: ignore[union-attr]
                'val_loss': logs.get('val_loss') if logs else None, # Example, can store more
                'best_score_checkpoint': self.best_score # Store the best score known at this checkpoint
            }
            if self.trainer.lr_scheduler is not None:  # type: ignore[union-attr]
                checkpoint_data['scheduler_state_dict'] = self.trainer.lr_scheduler.state_dict()  # type: ignore[union-attr]
            if self.trainer.scaler is not None:  # type: ignore[union-attr]
                checkpoint_data['scaler_state_dict'] = self.trainer.scaler.state_dict()  # type: ignore[union-attr]
            torch.save(checkpoint_data, path)


class TensorBoardLogger(Callback):
    def __init__(self, log_dir: str = './logs'):
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        logs = logs or {}
        if epoch is None:
            return
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
        # Log learning rate using get_last_lr() instead of relying on verbose output
        if self.trainer and self.trainer.lr_scheduler:
            if hasattr(self.trainer.lr_scheduler, 'get_last_lr'):
                for i, lr in enumerate(self.trainer.lr_scheduler.get_last_lr()):
                    self.writer.add_scalar(f'learning_rate/group_{i}', lr, epoch)
        # Also log from optimizer directly as a fallback
        if self.trainer and self.trainer.optimizer:
            for i, param_group in enumerate(self.trainer.optimizer.param_groups):
                self.writer.add_scalar(f'learning_rate/group_{i}', param_group['lr'], epoch)
    def on_train_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        self.writer.close()


class TQDMProgressBar(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_bar: Optional[tqdm] = None
        self.batch_bar: Optional[tqdm] = None

    def on_train_begin(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        print(f"🚀 Starting Training on {self.trainer.device}...")  # type: ignore[union-attr]

    def on_epoch_begin(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        if epoch is None or self.trainer is None:
            return
        self.epoch_bar = tqdm(total=self.trainer.num_epochs, initial=epoch, unit="epoch", desc=f"Epoch {epoch+1}/{self.trainer.num_epochs}", position=0, leave=True)
        if self.trainer.train_loader is not None:
            self.batch_bar = tqdm(total=len(self.trainer.train_loader), unit="batch", desc="Training", position=1, leave=False)

    def on_batch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        logs = logs or {}
        if self.batch_bar is not None:
            self.batch_bar.update(1)
            self.batch_bar.set_postfix(logs, refresh=True)

    def on_epoch_end(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        logs = logs or {}
        if self.batch_bar is not None:
            self.batch_bar.close()
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
            self.epoch_bar.set_postfix(logs, refresh=True)
        if epoch is not None and self.trainer is not None and (epoch == self.trainer.num_epochs - 1 or self.trainer.stop_training):
            if self.epoch_bar is not None:
                self.epoch_bar.close()
            print('✨ Training Completed'.center(100, '-'))

# --- Main Trainer Class ---
class AdvancedModelTrainer(Generic[T]):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[Any] = None,  # Use Any instead of private _LRScheduler
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True, # Automatic Mixed Precision
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None, # e.g., 1.0
        callbacks: Optional[List[Callback]] = None,
        random_seed: Optional[int] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics or {}
        self.callbacks = callbacks if callbacks is not None else []

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.device = self._setup_device(device)
        self.model.to(self.device)

        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None  # type: ignore[call-arg]

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        for metric_name in self.metrics.keys():
            self.history[f"train_{metric_name}"] = []
            self.history[f"val_{metric_name}"] = []

        self.stop_training = False
        self.num_epochs = 0
        self.train_loader: Optional[DataLoader[T]] = None
        self.val_loader: Optional[DataLoader[T]] = None

        for cb in self.callbacks: # Link trainer to callbacks
            cb.set_trainer(self)

    def _setup_device(self, device_option: Optional[Union[str, torch.device]]) -> torch.device:
        if device_option:
            return torch.device(device_option)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _call_callbacks(self, event_name: str, epoch: Optional[int] = None, batch_idx: Optional[int] = None, logs: LogsType = None) -> None:
        for cb in self.callbacks:
            method = getattr(cb, event_name)
            # Determine which parameters to pass based on the event type
            if event_name in ['on_train_begin', 'on_train_end']:
                method(logs=logs)  # Training lifecycle events don't need epoch
            elif 'batch' in event_name:  # Batch-related events need batch_idx
                method(epoch=epoch, batch_idx=batch_idx, logs=logs)
            else:  # Epoch-related events only need epoch
                method(epoch=epoch, logs=logs)

    def _parse_batch(self, batch: Any) -> Tuple[Any, Any]:
        """
        Parses a batch from the DataLoader. Assumes batch is (inputs, targets, *optional_other_data).
        Inputs can be a single tensor or a tuple/list of tensors.
        Targets are expected to be a single tensor or tuple/list of tensors.
        Override this method if your DataLoader yields batches in a different format.
        """
        if len(batch) == 2: # inputs, targets
            inputs, targets = batch
        elif len(batch) >= 3: # inputs, targets, name/other_info
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError(f"Batch format not understood. Expected (inputs, targets, *optional) but got {len(batch)} elements.")

        # Recursively move tensors to device, supporting nested structures
        inputs = self._to_device(inputs)
        targets = self._to_device(targets)

        return inputs, targets

    def _to_device(self, data: Any) -> Any:
        """
        Recursively moves data to the configured device.
        Handles tensors, lists, tuples, and dictionaries containing tensors.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._to_device(item) for item in data)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        else:
            # For other types (e.g., scalars, numpy arrays), return as is
            # If needed, you could add conversion for numpy arrays here
            return data

    def _train_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        inputs, targets = self._parse_batch(batch)
        with torch.autocast(device_type='cuda', enabled=self.use_amp):  # Updated autocast syntax
            # Always treat inputs as a tuple of tensors to unpack
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs,)
            outputs = self.model(*inputs)

            try:
                loss = self.criterion(outputs, targets)
            except Exception as e:
                # Basic shape/dtype debugging
                if isinstance(outputs, (list, tuple)) and isinstance(targets, (list, tuple)):
                    for idx, (an_output, an_y) in enumerate(zip(outputs, targets)):
                        if an_output.shape != an_y.shape:
                            print(f'❌ Output shape = {an_output.shape}, Target shape = {an_y.shape} at index {idx}')
                        if an_output.dtype != an_y.dtype:
                            print(f'❌ Output dtype = {an_output.dtype}, Target dtype = {an_y.dtype} at index {idx}')
                elif not isinstance(outputs, (list, tuple)) and not isinstance(targets, (list, tuple)):
                     if outputs.shape != targets.shape:
                        print(f'❌ Output shape = {outputs.shape}, Target shape = {targets.shape}')
                     if outputs.dtype != targets.dtype:
                        print(f'❌ Output dtype = {outputs.dtype}, Target dtype = {targets.dtype}')
                raise e

        # Normalize loss for gradient accumulation
        loss_scaled_for_accum = loss / self.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss_scaled_for_accum).backward()  # type: ignore[union-attr]
        else:
            loss_scaled_for_accum.backward()

        batch_metrics = {"loss": loss.item()}
        with torch.no_grad(): # Metrics shouldn't affect gradients
            for name, func in self.metrics.items():
                batch_metrics[name] = func(outputs, targets).item()

        return outputs, loss, batch_metrics

    def _eval_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        inputs, targets = self._parse_batch(batch)

        with torch.no_grad():
            if isinstance(inputs, (list, tuple)):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            batch_metrics = {"loss": loss.item()}
            for name, func in self.metrics.items():
                batch_metrics[name] = func(outputs, targets).item()

        return outputs, loss, batch_metrics

    def _train_one_epoch(self, epoch_idx: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        metric_totals: Dict[str, float] = {name: 0.0 for name in self.metrics.keys()}
        num_samples = 0

        self.optimizer.zero_grad() # Initialize gradients once for accumulation

        for batch_idx, batch in enumerate(self.train_loader):  # type: ignore[union-attr]
            self._call_callbacks('on_batch_begin', epoch=epoch_idx, batch_idx=batch_idx)

            _, _loss, batch_metrics = self._train_step(batch)  # Rename to indicate it's intentionally unused

            # Handle gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):  # type: ignore[union-attr]
                if self.clip_grad_norm is not None and self.scaler: # AMP
                    self.scaler.unscale_(self.optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif self.clip_grad_norm is not None: # No AMP
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                else: # No clipping
                    if self.scaler: # AMP
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else: # No AMP
                        self.optimizer.step()
                self.optimizer.zero_grad()

            # Assuming inputs is either a tensor or the first element of a list/tuple of tensors
            batch_size = batch[0][0].size(0) if isinstance(batch[0], (list, tuple)) else batch[0].size(0)

            total_loss += batch_metrics["loss"] * batch_size
            for name in self.metrics.keys():
                metric_totals[name] += batch_metrics[name] * batch_size
            num_samples += batch_size

            log_batch_metrics = {f"train_{k}": v for k,v in batch_metrics.items()}
            self._call_callbacks('on_batch_end', epoch=epoch_idx, batch_idx=batch_idx, logs=log_batch_metrics)

        avg_loss = total_loss / num_samples
        avg_metrics = {f"train_{name}": total / num_samples for name, total in metric_totals.items()}
        avg_metrics["train_loss"] = avg_loss
        return avg_metrics

    def _evaluate_one_epoch(self, loader: DataLoader[T], prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metric_totals: Dict[str, float] = {name: 0.0 for name in self.metrics.keys()}
        num_samples = 0

        desc = "Validating" if prefix == "val" else "Testing"
        eval_bar = tqdm(loader, desc=desc, leave=False, position=1)

        for batch in eval_bar:
            _, _, batch_metrics = self._eval_step(batch) # We only need metrics here

            # Assuming inputs is either a tensor or the first element of a list/tuple of tensors
            batch_size = batch[0][0].size(0) if isinstance(batch[0], (list, tuple)) else batch[0].size(0)

            total_loss += batch_metrics["loss"] * batch_size
            for name in self.metrics.keys():
                metric_totals[name] += batch_metrics[name] * batch_size
            num_samples += batch_size

            eval_bar.set_postfix({f"{prefix}_loss": total_loss / num_samples if num_samples > 0 else 0.0}, refresh=True)

        avg_loss = total_loss / num_samples
        avg_metrics = {f"{prefix}_{name}": total / num_samples for name, total in metric_totals.items()}
        avg_metrics[f"{prefix}_loss"] = avg_loss
        return avg_metrics

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader[T],
        val_loader: Optional[DataLoader[T]] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        self.num_epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        start_epoch = 0

        if resume_from_checkpoint:
            try:
                checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', -1) + 1 # Resume from next epoch

                if self.lr_scheduler and 'scheduler_state_dict' in checkpoint:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

                # Restore best score for callbacks like ModelCheckpoint and EarlyStopping if available
                best_score_chkpt = checkpoint.get('best_score_checkpoint')
                for cb in self.callbacks:
                    if hasattr(cb, 'best_score') and best_score_chkpt is not None:
                        cb.best_score = best_score_chkpt  # type: ignore[attr-defined]
                    if isinstance(cb, EarlyStopping) and 'epoch' in checkpoint: # Adjust patience based on last saved epoch
                        cb.best_epoch = checkpoint.get('epoch') # Assume best was at this saved epoch

                print(f"Resumed training from epoch {start_epoch} using checkpoint '{resume_from_checkpoint}'")
            except FileNotFoundError:
                print(f"Checkpoint file '{resume_from_checkpoint}' not found. Starting training from scratch.")
            except Exception as e:
                print(f"Could not load checkpoint '{resume_from_checkpoint}': {e}. Starting training from scratch.")


        self._call_callbacks('on_train_begin')
        self.stop_training = False # Reset stop_training flag

        for epoch_idx in range(start_epoch, epochs):
            if self.stop_training:
                print("Training stopped prematurely by a callback or external signal.")
                break

            epoch_logs: Dict[str, float] = {}
            self._call_callbacks('on_epoch_begin', epoch=epoch_idx)

            # Training phase
            train_metrics = self._train_one_epoch(epoch_idx)
            epoch_logs.update(train_metrics)

            # Validation phase
            if self.val_loader:
                self._call_callbacks('on_validation_begin', epoch=epoch_idx)
                val_metrics = self._evaluate_one_epoch(self.val_loader, prefix="val")
                epoch_logs.update(val_metrics)
                self._call_callbacks('on_validation_end', epoch=epoch_idx, logs=val_metrics)

            # Update history
            for key, value in epoch_logs.items():
                if key not in self.history: self.history[key] = []
                self.history[key].append(value)

            # LR Scheduler step (some schedulers step per epoch, some per batch)
            # Common practice for ReduceLROnPlateau is to step with validation metric
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in epoch_logs: # Or any other monitored metric
                        self.lr_scheduler.step(epoch_logs['val_loss'])
                    else:
                        print("Warning: ReduceLROnPlateau scheduler needs 'val_loss' or monitored metric in epoch_logs.")
                else: # For other schedulers like StepLR, CosineAnnealingLR
                    self.lr_scheduler.step()

            self._call_callbacks('on_epoch_end', epoch=epoch_idx, logs=epoch_logs)

        self._call_callbacks('on_train_end', logs=self.history)
        return self.history

    def evaluate(self, test_loader: DataLoader[T], prefix: str = "test") -> Dict[str, float]:
        self.model.to(self.device) # Ensure model is on correct device
        return self._evaluate_one_epoch(test_loader, prefix=prefix)

    def predict(self, data_loader: DataLoader[T]) -> List[torch.Tensor]:
        self.model.eval()
        self.model.to(self.device)
        predictions = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting", leave=False):
                inputs, _ = self._parse_batch(batch) # We only need inputs for prediction
                if isinstance(inputs, (list, tuple)):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                predictions.append(outputs.cpu())
        return predictions


if __name__ == '__main__':
    # 0. Seed for reproducibility
    SEED = 42

    # 1. Dummy Data and DataLoader
    # Let's create a simple regression problem
    X_train_np = np.random.rand(1000, 10).astype(np.float32)
    y_train_np = (X_train_np @ np.random.rand(10, 1).astype(np.float32) + np.random.randn(1000,1).astype(np.float32) * 0.1).astype(np.float32)

    X_val_np = np.random.rand(200, 10).astype(np.float32)
    y_val_np = (X_val_np @ np.random.rand(10, 1).astype(np.float32) + np.random.randn(200,1).astype(np.float32) * 0.1).astype(np.float32)

    X_test_np = np.random.rand(200, 10).astype(np.float32)
    y_test_np = (X_test_np @ np.random.rand(10, 1).astype(np.float32) + np.random.randn(200,1).astype(np.float32) * 0.1).astype(np.float32)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_val = torch.from_numpy(X_val_np)
    y_val = torch.from_numpy(y_val_np)
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)

    # Create datasets with sample indices as the third element
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Custom collate function to add the sample info as third element
    def train_collate_fn(batch: List[Any]) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, List[str]]:
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        # Add sample indices as the third element (name/other_info)
        indices = [f"sample_{i}" for i in range(len(batch))]
        return (inputs, ), targets, indices

    def val_collate_fn(batch: List[Any]) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, List[str]]:
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        indices = [f"val_sample_{i}" for i in range(len(batch))]
        return (inputs, ), targets, indices

    def test_collate_fn(batch: List[Any]) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, List[str]]:
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        indices = [f"test_sample_{i}" for i in range(len(batch))]
        return (inputs, ), targets, indices

    # Create data loaders with the collate functions
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=val_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=test_collate_fn)

    # 2. Simple Model
    class SimpleNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 640000)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(640000, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Simple forward function that takes a single input tensor
            # The model is called with model(*x) where x is the first element from the batch
            # Which is just the input tensor itself
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleNet(input_dim=10, output_dim=1)

    # Example of using the older BaseModel from crocodile.deeplearning_torch
    from crocodile.deeplearning_torch import BaseModel
    mm = BaseModel(model=model, loss=nn.MSELoss(), optimizer=optim.Adam(lr=0.005, params=model.parameters()), metrics=[])
    mm.fit(epochs=50, train_loader=train_loader, test_loader=test_loader)
    import polars as pl
    import plotly.express as px
    df = pl.DataFrame(mm.history[0]).with_row_index(name="epoch").to_pandas()
    fig = px.line(data_frame=df, x='epoch', y=['train_loss', 'test_loss'], title='Training and Validation Loss')
    fig.show()
    # raise ValueError

    # Create a new model for our AdvancedModelTrainer
    model = SimpleNet(input_dim=10, output_dim=1)
    # 3. Loss, Optimizer, Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # AdamW is often better
    # Scheduler that reduces LR when validation loss plateaus - remove verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 4. Metrics (optional)
    def r_squared(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return torch.tensor(1.0) - (ss_res / (ss_tot + 1e-8)) # add epsilon for stability

    metrics_dict = {"r2": r_squared}
    # 5. Callbacks
    early_stopper = EarlyStopping(monitor='val_loss', patience=30, verbose=True, restore_best_weights=False)
    # {epoch:02d} means epoch number with 2 digits, zero-padded. {val_loss:.2f} means val_loss formatted to 2 decimal places.
    model_checkpointer = ModelCheckpoint(filepath='./checkpoints/best_model_epoch_{epoch:02d}_valloss_{val_loss:.4f}.pth',
                                         monitor='val_loss', mode='min', save_best_only=True, verbose=True)
    tensorboard_logger = TensorBoardLogger(log_dir='./tb_logs/experiment1')
    progress_bar = TQDMProgressBar()

    callbacks_list = [early_stopper, model_checkpointer, tensorboard_logger, progress_bar]

    # 6. Initialize Trainer
    trainer = AdvancedModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        metrics=metrics_dict,
        # device="cpu", # Force CPU for testing if no GPU
        use_amp=True, # Will automatically be False if on CPU or CUDA not available for AMP
        gradient_accumulation_steps=2, # Simulate batch_size = 32 * 2 = 64
        clip_grad_norm=1.0,
        callbacks=callbacks_list,
        random_seed=SEED
    )

    # 7. Start Training
    print(f"Using device: {trainer.device}")
    print(f"Using AMP: {trainer.use_amp}")

    history = trainer.fit(epochs=100, train_loader=train_loader, val_loader=val_loader)
    # To resume:
    # history = trainer.fit(epochs=100, train_loader=train_loader, val_loader=val_loader, resume_from_checkpoint='./checkpoints/last_checkpoint.pth')


    # 8. Evaluate on Test Set (using the best model restored by EarlyStopping or loaded manually)
    print("\nEvaluating on Test Set:")
    test_results = trainer.evaluate(test_loader)
    print(f"Test Results: {test_results}")

    # 9. Make Predictions
    print("\nMaking predictions on Test Set (first batch):")
    # Create a small loader for prediction example
    first_batch_test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=test_collate_fn)
    predictions = trainer.predict(first_batch_test_loader) # Returns a list of tensors (one per batch)

    print(f"Shape of first prediction batch: {predictions[0].shape}")
    print(f"First 5 predictions: \n{predictions[0]}")

    # Print history
    print("\nTraining History:")
    import polars as pl
    print(pl.DataFrame(history).to_pandas())
    # for key, values in history.items():
    #     if values: # Only print if there's data
    #         print(f"{key}: {[f'{v:.4f}' for v in values]}")
