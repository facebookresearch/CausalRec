# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import os
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
import os  # For setting PYTHONHASHSEED


from src.models.utils import AlphaRise, FilteringMlFlowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


class SaveAndTestAtEpoch(Callback):
    def __init__(self, save_epoch, save_path, dataset_collection, args):
        super().__init__()
        self.save_epoch = save_epoch
        self.save_path = save_path
        self.has_saved = False
        self.dataset_collection = dataset_collection
        self.args = args

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.save_epoch and not self.has_saved:
            # Save the model checkpoint
            trainer.save_checkpoint(self.save_path)
            self.has_saved = True
            pl.utilities.rank_zero_info(f"Model checkpoint saved at epoch {self.save_epoch} to {self.save_path}")

            # Run testing
            #pl.utilities.rank_zero_info(f"Running test at epoch {self.save_epoch}")

            # Create validation dataloader
            #val_dataloader = DataLoader(
            #    self.dataset_collection.val_f,
            #    batch_size=self.args.dataset.val_batch_size,
            #    shuffle=False
            #)
            # Run testing
            #trainer.test(pl_module, test_dataloaders=val_dataloader)
            #print("done testing")


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for CT (Causal Transformer)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('
' + OmegaConf.to_yaml(args, resolve=True))
    os.environ['PYTHONHASHSEED'] = str(args.exp.seed)
    
    # Initialisation of data
    # Set random seed
    seed = args.exp.seed
    #seed_everything(seed)
    seed_everything(args.exp.seed, workers=True)

    # Get balancing switch epoch
    save_epoch = args.exp.balancing_switch_epoch
    # Construct the save path using configuration value for checkpoint directory
    save_dir = args.exp.checkpoint_dir  # Get the save directory from the config
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists



    # Construct save path including random seed
    # Save the checkpoint in the desired directory
    save_path = os.path.join(save_dir, f'checkpoint_seed_{seed}_epoch_{save_epoch}.ckpt')
    save_path = to_absolute_path(save_path)  # Convert to absolute path to avoid Hydra issues

    #save_path = f'./checkpoint_seed_{seed}_epoch_{save_epoch}.ckpt'

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Get the model class from args
    model_class = get_class(args.model.multi._target_)

    # Train_callbacks
    multimodel_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        multimodel_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None
    
    # Check if checkpoint exists
    if os.path.exists(save_path):
        pl.utilities.rank_zero_info(f"Checkpoint {save_path} exists. Loading model from checkpoint and starting fine-tuning.")
        # ============================== Initialisation & Training of multimodel ==============================

        # Load the model from the checkpoint
        #multimodel = model_class.load_from_checkpoint(save_path, args=args, dataset_collection=dataset_collection)
        multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)

        checkpoint = torch.load(save_path, map_location='cpu')

        multimodel.load_state_dict(checkpoint['state_dict'])

        # Set current_balancing_method to final_balancing_method
        multimodel.current_balancing_method = args.exp.final_balancing_method
        
        current_epoch = checkpoint['epoch']  # Extracting the current epoch
        global_step = checkpoint.get('global_step', 0)  # You can also get the global step if needed

        # Initialize trainer
        multimodel_trainer = Trainer(
            gpus=eval(str(args.exp.gpus)),
            logger=mlf_logger,
            max_epochs=args.exp.max_epochs,
            callbacks=multimodel_callbacks,
            terminate_on_nan=True,
            gradient_clip_val=args.model.multi.max_grad_norm,
            deterministic=True, 
            benchmark=False
            #resume_from_checkpoint=save_path
        )
        multimodel_trainer.fit_loop.epoch_progress.current.completed = current_epoch
        multimodel_trainer.fit_loop.epoch_loop._batches_that_stepped = global_step

    else:
        pl.utilities.rank_zero_info("Checkpoint does not exist. Starting training from scratch.")
        multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
        if args.model.multi.tune_hparams:
            multimodel.finetune(resources_per_trial=args.model.multi.resources_per_trial)

        # Add the custom callback to multimodel_callbacks
        multimodel_callbacks += [SaveAndTestAtEpoch(save_epoch=save_epoch, save_path=save_path, dataset_collection=dataset_collection, args=args)]


        multimodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), 
                                     logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                     callbacks=multimodel_callbacks, terminate_on_nan=True,
                                     gradient_clip_val=args.model.multi.max_grad_norm,
                                     deterministic=True,
                                     benchmark=False)
    
    print("done before")
    multimodel_trainer.fit(multimodel)
    print("done after")

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    multimodel_trainer.test(multimodel, test_dataloaders=val_dataloader)
    # multimodel.visualize(dataset_collection.val_f, index=0, artifacts_path=artifacts_path)
    val_rmse_orig, val_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = multimodel.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
                                                                                              one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_orig': val_rmse_orig
    }
    decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

    mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
    results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()
