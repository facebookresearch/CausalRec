# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, Callback

import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from src.models.utils import AlphaRise, FilteringMlFlowLogger
import os  # For setting PYTHONHASHSEED

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


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for models with encoder-decoder structure: CRN, EDCT (Encoder-Decoder Causal Transformer)
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
    save_path_encoder = os.path.join(save_dir, f'checkpoint_encoder_seed_{seed}_epoch_{save_epoch}.ckpt')
    save_path_encoder = to_absolute_path(save_path_encoder)  # Convert to absolute path to avoid Hydra issues

    save_path_decoder = os.path.join(save_dir, f'checkpoint_decoder_seed_{seed}_epoch_{save_epoch}.ckpt')
    save_path_decoder = to_absolute_path(save_path_decoder)  # Convert to absolute path to avoid Hydra issues


    # Initialisation of data
    #seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_encoder()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    encoder_callbacks, decoder_callbacks = [AlphaRise(rate=args.exp.alpha_rate)], [AlphaRise(rate=args.exp.alpha_rate)]

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=['encoder', 'decoder'],
                                           experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # Check if checkpoint exists
    if os.path.exists(save_path_encoder):
        pl.utilities.rank_zero_info(f"Checkpoint {save_path_encoder} exists. Loading model from encoder checkpoint and starting fine-tuning.")
        #pl.utilities.rank_zero_info(f"Checkpoint {save_path_decoder} exists. Loading model from decoder checkpoint and starting fine-tuning.")
        # ============================== Initialisation & Training of multimodel ==============================
    
        encoder = instantiate(args.model.encoder, args, dataset_collection, _recursive_=False)
        checkpoint_encoder = torch.load(save_path_encoder, map_location='cpu')
        encoder.load_state_dict(checkpoint_encoder['state_dict'])
        
        # Set current_balancing_method to final_balancing_method
        encoder.current_balancing_method = args.exp.final_balancing_method
        
        current_epoch = checkpoint_encoder['epoch']  # Extracting the current epoch
        global_step = checkpoint_encoder.get('global_step', 0)  # You can also get the global step if needed

        # Initialize trainer
        encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                              callbacks=encoder_callbacks, terminate_on_nan=True, deterministic=True, benchmark=False)
    
    

        encoder_trainer.fit_loop.epoch_progress.current.completed = current_epoch
        encoder_trainer.fit_loop.epoch_loop._batches_that_stepped = global_step
        
    else:
        pl.utilities.rank_zero_info("Encoder checkpoint does not exist. Starting training from scratch.")
    
        encoder = instantiate(args.model.encoder, args, dataset_collection, _recursive_=False)
        if args.model.encoder.tune_hparams:
            encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

        # Add the custom callback to multimodel_callbacks
        encoder_callbacks += [SaveAndTestAtEpoch(save_epoch=save_epoch, save_path=save_path_encoder, dataset_collection=dataset_collection, args=args)]

    
        encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                  callbacks=encoder_callbacks, terminate_on_nan=True, deterministic=True, benchmark=False)
    print("done before")    
    encoder_trainer.fit(encoder)
    print("done after")

    encoder_results = {}

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    encoder_trainer.test(encoder, test_dataloaders=val_dataloader)
    val_rmse_orig, val_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = encoder.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
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
        test_rmse_orig, test_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.test_f)
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


    # ============================== Initialisation & Training of decoder ==============================
    if args.model.train_decoder:
        if args.exp.logging:
            mlf_logger.filter_submodel = 'encoder'  # Disable Logging to Mlflow
        
        if os.path.exists(save_path_decoder) and args.exp.final_balancing_method != 'subgroup':
            #pl.utilities.rank_zero_info(f"Checkpoint {save_path_encoder} exists. Loading model from encoder checkpoint and starting fine-tuning.")
            pl.utilities.rank_zero_info(f"Checkpoint {save_path_decoder} exists. Loading model from decoder checkpoint and starting fine-tuning.")
            # ============================== Initialisation & Training of multimodel ==============================
            decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)

            checkpoint_decoder = torch.load(save_path_decoder, map_location='cpu')
            decoder.load_state_dict(checkpoint_decoder['state_dict'])
        
            # Set current_balancing_method to final_balancing_method
            decoder.current_balancing_method = args.exp.final_balancing_method
        
            current_epoch = checkpoint_decoder['epoch']  # Extracting the current epoch
            global_step = checkpoint_decoder.get('global_step', 0)  # You can also get the global step if needed

            decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                  callbacks=decoder_callbacks, terminate_on_nan=True)
            
            decoder_trainer.fit_loop.epoch_progress.current.completed = current_epoch
            decoder_trainer.fit_loop.epoch_loop._batches_that_stepped = global_step

        else:
            pl.utilities.rank_zero_info("Decoder Checkpoint does not exist. Starting training from scratch.")
            decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)

            if args.model.decoder.tune_hparams:
                decoder.finetune(resources_per_trial=args.model.decoder.resources_per_trial)

            decoder_callbacks += [SaveAndTestAtEpoch(save_epoch=save_epoch, save_path=save_path_decoder, dataset_collection=dataset_collection, args=args)]

            decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                      callbacks=decoder_callbacks, terminate_on_nan=True)
        print("done before")
        decoder_trainer.fit(decoder)
        print("done after")

        # Validation factual rmse
        val_dataloader = DataLoader(dataset_collection.val_f, batch_size=10 * args.dataset.val_batch_size, shuffle=False)
        decoder_trainer.test(decoder, test_dataloaders=val_dataloader)
        # decoder.visualize(dataset_collection.val_f, index=20, artifacts_path=artifacts_path)
        val_rmse_orig, val_rmse_all = decoder.get_normalised_masked_rmse(dataset_collection.val_f)
        logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

        test_rmses = {}
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
            test_rmses = decoder.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, 'test_f'):  # Test n_step_factual rmse
            test_rmses = decoder.get_normalised_n_step_rmses(dataset_collection.test_f)
        test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
        decoder_results = {
            'decoder_val_rmse_all': val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig
        }
        decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

        mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
        results.update(decoder_results)

    return results


if __name__ == "__main__":
    main()
