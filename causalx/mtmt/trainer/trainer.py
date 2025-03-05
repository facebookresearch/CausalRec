import logging
import os
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
        self.loss_weights = torch.tensor(loss_config.get("loss_weight_list", None), dtype=torch.float32)  # loss weight
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
        
        self.da_method = loss_config.get("da_method", "mmd")  # Get method from config
        self.da_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> tuple[torch.optim.Optimizer, TLRScheduler | None]:
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                module.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                module.parameters(), 
                lr=self.lr
            )
        lr_scheduler = None
        return optimizer, lr_scheduler

    def compute_loss(self, state: State, data: Sample) -> tuple[torch.Tensor, Any]:
        targets = data["outcome"] # [batch x n_outcomes]
        trt = data["trt"] # [batch]
        if self.loss_type == "bce_dann":
            outputs, domain_preds = self.module(data)
        elif self.loss_type.startswith("bce_PLE2D_DA"):  # Handle both PLE2D_DA and PLE2D_DA_ctcvr
            if self.loss_type.endswith("ctcvr"):
                outputs_raw, treatment_shared_outputs = self.module(data)
                device = outputs_raw[0].device  # Access device from the first tensor
                M, N = outputs_raw.shape[0], outputs_raw.shape[2]
                outputs = torch.zeros(M, 2, N, device=device)
                outputs[:, 0, :] = outputs_raw[:, 0, :]
                outputs[:, 1, :] = outputs_raw[:, 0, :] * outputs_raw[:, 1, :]
            else:
                outputs, treatment_shared_outputs = self.module(data)
        elif self.loss_type.endswith("ctcvr"):
            outputs_raw = self.module(data)
            device = outputs_raw.device
            M, N = outputs_raw.shape[0], outputs_raw.shape[2]
            outputs = torch.zeros(M, 2, N, device=device)
            outputs[:, 0, :] = outputs_raw[:, 0, :]
            outputs[:, 1, :] = outputs_raw[:, 0, :] * outputs_raw[:, 1, :]
        else:
            outputs = self.module(data)
        if outputs.shape[2] > 1: # multi-head type: [batch x n_outcomes x m_treatments];
            trt_e = trt.unsqueeze(1).expand(-1, outputs.shape[1])
            outputs = outputs.gather(2, trt_e.unsqueeze(2)).squeeze(2) 
        else: # s-learner type: [batch x n_outcomes x 1]
            outputs = outputs.squeeze(2) 
        if targets.shape[1] != outputs.shape[1]:
            raise ValueError(
                f"label_num shape: {targets.shape} is different from model output shape: {outputs.shape}"
            )

        loss_class = None
        if self.loss_type == "l2":
            loss_class = F.mse_loss
        elif self.loss_type == "l1":
            loss_class = F.l1_loss
        elif self.loss_type == "smooth_l1":
            loss_class = F.smooth_l1_loss
        elif self.loss_type.startswith("bce"):
            loss_class = F.binary_cross_entropy
        else:
            raise ValueError(
                f"unsupported loss type: {self.loss_type}"
            )
        
        full_losses = loss_class(outputs, targets, reduction='none')
        mean_losses = full_losses.mean(dim=0) # shape [n_outcomes]
        weighted_losses = torch.dot(mean_losses, self.loss_weights)
        main_loss = weighted_losses.sum()

        if self.loss_type == "bce_dann":
            # Domain loss: cross entropy with treatment as classes
            domain_loss = F.cross_entropy(domain_preds, trt)
            total_loss = main_loss + domain_loss
            return total_loss, outputs
        elif self.loss_type.startswith("bce_PLE2D_DA"):  # Handle both PLE2D_DA and PLE2D_DA_ctcvr
            # treatment_shared_outputs shape: [num_treatment_shared_experts, batch_size, hidden_dim]
            num_experts = treatment_shared_outputs.shape[0]
            
            da_losses = []
            batch_size = outputs.shape[0]  # Get batch size for normalization
            
            # Process each expert independently
            for expert_idx in range(num_experts):
                expert_outputs = treatment_shared_outputs[expert_idx]  # [batch, hidden_dim]
                
                # Group outputs by treatment for this expert
                treatment_groups = {}
                for i in range(len(trt)):
                    t = trt[i].item()
                    if t not in treatment_groups:
                        treatment_groups[t] = []
                    treatment_groups[t].append(expert_outputs[i])
                
                # Skip if we don't have enough samples for comparison
                if len(treatment_groups) < 2:
                    continue
                    
                # Combine all data for this expert
                all_data = torch.cat([torch.stack(treatment_groups[t]) for t in treatment_groups], dim=0)
                
                # Compare each treatment group against the combined data
                expert_loss = 0
                num_comparisons = 0
                for t in treatment_groups:
                    t_data = torch.stack(treatment_groups[t])
                    #print(f"self.da_loss(t_data, all_data): {self.da_loss(t_data, all_data)}")
                    expert_loss += self.da_loss(t_data, all_data) / batch_size
                    num_comparisons += 1
                
                if num_comparisons > 0:
                    da_losses.append(expert_loss / num_comparisons)
            
            if da_losses:
                da_loss = torch.stack(da_losses).mean()
                # Add gradient clipping to prevent explosion
                da_loss = torch.clamp(da_loss, max=self.loss_th)
                print(f"Main loss: {main_loss:.4f}, DA loss: {da_loss:.4f}, Total loss: {main_loss + self.da_weight * da_loss:.4f}")
                total_loss = main_loss + self.da_weight * da_loss
            else:
                total_loss = main_loss
                
            return total_loss, outputs
        else:
            return main_loss, outputs

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

        # Load checkpoint
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
            "Finished Eval Epoch: {} \tEval Epoch Loss: {:.6f}".format(  # noqa
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
            "Finished Train Epoch: {} \tTrain Epoch Loss: {:.6f}".format(  # noqa
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
        train_config: dict[str, Any],
        device: torch.device
    ):  
        self.unit = unit
        self.train_config = train_config
        self.device = device

    def train(
        self,
        train_dataset,
        eval_dataset,
    ) -> tuple[list[str], list[float]]:
        batch_size = self.train_config["batch_size"]
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
            max_epochs=self.train_config["max_epochs"],
            evaluate_every_n_epochs=self.train_config["evaluate_every_n_epochs"],
            callbacks=[],
        )
        losses = [self.unit.train_loss, self.unit.eval_loss]
        # Get saved checkpoint paths
        self.checkpoint_paths = getattr(self.unit, 'checkpoint_paths', [])
        
        return self.checkpoint_paths, losses
        
    def predict(
        self,
        dataset: Dataset,
    ):  
        module = self.unit.module
        module.eval()
        samples = dataset[:]
        with torch.no_grad():
            pred = module(samples)
        return pred
    
    @staticmethod
    def prepare_dataloader(
        dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device
    ) -> DataLoader:
        """Instantiate DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )