# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import logging
import os
import pdb
from typing import Any, Union, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Union, Optional
from torchtnt.framework.auto_unit import AutoUnit, TrainStepResults
from torchtnt.utils.loggers import TensorBoardLogger
from torchtnt.framework.state import State
from torchtnt.framework.fit import fit
from torchtnt.utils.lr_scheduler import TLRScheduler
from torcheval.metrics import MeanSquaredError
from torchtnt.utils.env import init_from_env
import torch.nn.functional as F
from geomloss import SamplesLoss
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True

Sample = dict[str, Union[torch.Tensor, list[torch.Tensor]]]

class TrainerUnit(AutoUnit[Sample]):
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        tb_logger: TensorBoardLogger,
        log_every_n_steps: int,
        lr: float,
        optimizer_name: str,
        weight_decay: float,
        loss_config: dict[str, Any],
        checkpoint_config: Optional[dict[str, Any]] = None,
        **kwargs: dict[str, Any],  # kwargs to be passed to AutoUnit
    ):  
        kwargs["module"] = module
        super().__init__(**kwargs)
        self.tb_logger = tb_logger
        # Checkpoint configuration with default values
        self.checkpoint_config = checkpoint_config or {
            "save_dir": "checkpoints",
            "save_interval": 5,
            "max_keep": 3
        }
        
        # Ensure save directory exists
        os.makedirs(self.checkpoint_config["save_dir"], exist_ok=True)
        
        # create epoch_loss metrics to compute the epoch_loss of training and evaluation
        self.train_epoch_loss = 0
        self.eval_epoch_loss = 0
        self.num_batch_train = 0
        self.num_batch_eval = 0
        self.log_every_n_steps = log_every_n_steps
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.loss_type = loss_config["loss_type"]
        weight_list = loss_config.get("loss_weight_list")
        self.loss_weights = (torch.tensor(weight_list, dtype=torch.float32) 
                           if weight_list is not None else None)
        if "device" in kwargs:
            self.loss_weights = self.loss_weights.to(kwargs["device"])
        self.loss_th = loss_config.get("loss_th", 1)
        self.da_weight = loss_config.get("da_weight", 0.1)
        self.train_loss = 0
        self.eval_loss = 0

        # Checkpoint tracking
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.checkpoint_paths = []
        self.optimizer = None  # Should be initialized before using in save_checkpoint

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> tuple[torch.optim.Optimizer, TLRScheduler | None]:
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                module.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:  # default to SGD
            optimizer = torch.optim.SGD(
                module.parameters(), 
                lr=self.lr
            )
        lr_scheduler = None
        return optimizer, lr_scheduler

    def compute_loss(self, state: State, data: Sample) -> tuple[torch.Tensor, Any]:
        targets = data["outcome"]  # [batch x 1]
        trt = data["trt"]  # [batch]
        outputs = self.module(data)
        # Validate shapes
        if len(outputs.shape) != 2:
            raise ValueError(f"Expected 2D output tensor (batch, n_treatments), got shape {outputs.shape}")
        if outputs.shape[1] != 2:  # Assuming binary treatment
            raise ValueError(f"Expected 2 treatment outputs, got {outputs.shape[1]}")
        
        # Assume multi-head for treatment-specific outputs
        # Sanity check for treatment values
        unique_trt = torch.unique(trt)
        expected_treatments = torch.tensor([0, 1], device=unique_trt.device)  # Use same device as input
        if not torch.equal(unique_trt.sort().values, expected_treatments):
            raise ValueError(f"Treatment values must be 0 or 1, got {unique_trt}")
        
        trt_e = trt.unsqueeze(1)  # [batch x 1]
        
        outputs = outputs.gather(1, trt_e)
        loss_class = None
        if self.loss_type == "l2":
            loss_class = F.mse_loss
        elif self.loss_type == "l1":
            loss_class = F.l1_loss
        elif self.loss_type == "smooth_l1":
            loss_class = F.smooth_l1_loss
        elif self.loss_type.startswith("bce"):
            loss_class = F.binary_cross_entropy
            # Ensure outputs are in [0,1]
            outputs = torch.sigmoid(outputs)
            # Ensure targets are in [0,1]
            if not ((0 <= targets) & (targets <= 1)).all():
                raise ValueError("Targets must be in range [0,1] for BCE loss")
        else:
            raise ValueError(
                f"unsupported loss type: {self.loss_type}"
            )
        
        loss = loss_class(outputs, targets)
        return loss, outputs

    def save_checkpoint(self, epoch_loss: float) -> Optional[str]:
        """
        Save model checkpoint with optional best model tracking
        """

        save_dir = self.checkpoint_config.get('save_dir', 'checkpoints')
        max_keep = self.checkpoint_config.get('max_keep', 3)
        save_interval = self.checkpoint_config.get('save_interval', 5)

        print(f"Checkpoint saving attempt:")
        print(f"Current epoch: {self.current_epoch}")
        print(f"Save interval: {save_interval}")
        print(f"Save directory: {save_dir}")
        print(f"Checkpoint save condition met: {self.current_epoch % save_interval == 0}")

        # Only save checkpoint at specified intervals
        if self.current_epoch % save_interval != 0:
            return None

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)


        # Generate checkpoint filename
        checkpoint_filename = f'checkpoint_epoch_{self.current_epoch}.pt'
        checkpoint_path = os.path.join(save_dir, checkpoint_filename)

        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': epoch_loss,
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Manage checkpoint history
        self.checkpoint_paths.append(checkpoint_path)
        if len(self.checkpoint_paths) > max_keep:
            # Remove the oldest checkpoint
            oldest_checkpoint = self.checkpoint_paths.pop(0)
            try:
                os.remove(oldest_checkpoint)
                logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
            except OSError as e:
                logger.warning(f"Error removing old checkpoint: {e}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        if not hasattr(self, 'device'):
            self.device = next(self.module.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model state
        self.module.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state if available
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore epoch and loss information
        self.current_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def on_train_step_end(
        self,
        state: State,
        data: Sample,
        step: int,
        results: TrainStepResults,
    ) -> None:
        loss, outputs = results.loss, results.outputs
        self.train_epoch_loss += loss.item()
        self.num_batch_train += 1
        
        if step % self.log_every_n_steps == 0:
            self.tb_logger.log("train_loss", loss, step)

    def on_eval_step_end(
        self,
        state: State,
        data: Sample,
        step: int,
        loss: torch.Tensor,
        outputs: Any,
    ) -> None:
        self.eval_epoch_loss += loss.item()
        self.num_batch_eval +=1

    def on_eval_end(self, state: State) -> None:
        epoch = self.eval_progress.num_epochs_completed
        epoch_loss = self.eval_epoch_loss / self.num_batch_eval
        self.eval_loss = epoch_loss
        self.tb_logger.log("eval_epoch_loss", epoch_loss, epoch)
        print(
            "Finished Eval Epoch: {} 	Eval Epoch Loss: {:.6f}".format(  # noqa
                epoch,
                epoch_loss,
            )
        )
        self.eval_epoch_loss = 0
        self.num_batch_eval = 0

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        epoch = self.train_progress.num_epochs_completed
        epoch_loss = self.train_epoch_loss / self.num_batch_train
        self.tb_logger.log("train_epoch_loss", epoch_loss, epoch)
        print(
            "Finished Train Epoch: {} 	Train Epoch Loss: {:.6f}".format(  # noqa
                epoch,
                epoch_loss,
            )
        )
        self.train_loss = epoch_loss

        # Explicitly save checkpoint and print debug info
        print(f"Attempting to save checkpoint at epoch {epoch}")
        checkpoint_path = self.save_checkpoint(epoch_loss)
        print(f"Checkpoint path returned: {checkpoint_path}")
        
        # reset the metric every epoch
        self.train_epoch_loss = 0
        self.num_batch_train = 0

        # Increment current epoch
        self.current_epoch = epoch
        
    
class Trainer:
    def __init__(
        self,
        *,
        unit: TrainerUnit,
        trainer_config: dict[str, Any],
        device: torch.device
    ):  
        self.unit = unit
        self.trainer_config = trainer_config
        self.device = device

    def train(
        self,
        train_dataset,
        eval_dataset,
    ) -> tuple[list[str], list[float]]:
        try:
            batch_size = self.trainer_config["batch_size"]
            train_dataloader = self.prepare_dataloader(
                train_dataset, batch_size, True, self.device
            )
            eval_dataloader = self.prepare_dataloader(
                eval_dataset, batch_size, False, self.device
            )
            
            fit(
                self.unit,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_epochs=self.trainer_config["max_epochs"],
                evaluate_every_n_epochs=self.trainer_config["evaluate_every_n_epochs"],
                callbacks=[],
            )
            losses = [self.unit.train_loss, self.unit.eval_loss]
            # Get saved checkpoint paths
            self.checkpoint_paths = getattr(self.unit, 'checkpoint_paths', [])
            
            return self.checkpoint_paths, losses
        finally:
            # Clean up dataloaders
            del train_dataloader
            del eval_dataloader
            torch.cuda.empty_cache()  # If using GPU
        
    def predict(
        self,
        dataset: Dataset,
    ):  
        if not hasattr(self.unit, 'module') or self.unit.module is None:
            raise ValueError("No model available for prediction")
            
        module = self.unit.module
        module.eval()
        try:
            samples = dataset[:]
            with torch.no_grad():
                pred = module(samples)
            return pred
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    @staticmethod
    def prepare_dataloader(
        dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device
    ) -> DataLoader:
        """Instantiate DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Add num_workers
            pin_memory=device.type == "cuda"  # Add pin_memory for GPU
        )
