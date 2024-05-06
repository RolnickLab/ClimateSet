import wandb
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import emulator.src.utils.config_utils as cfg_utils
from emulator.src.utils.interface import get_model_and_data
from emulator.src.utils.utils import get_logger
from pytorch_lightning.profilers import PyTorchProfiler
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler, schedule
from codecarbon import EmissionsTracker



def run_model(config: DictConfig):
    seed_everything(config.seed, workers=True)
    log = get_logger(__name__)
    emissions_tracker_enabled = config.get('datamodule', {}).get('emissions_tracker', False)
    log.info("In run model")
    cfg_utils.extras(config)

    log.info("Running model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")


    
    emulator_model, data_module = get_model_and_data(config)
    log.info(f"Got model - {config.name}")
    c = datetime.now()
    # Displays Time
    current_time = c.strftime('%H:%M:%S')

    
    profiler = None
    checkpointing = True
    if config.get("pyprofile"):
        checkpointing = False
        profiler = PyTorchProfiler(dirpath="logs/profiles",filename=f"Pyprofile-{config.name}-Basetest-{current_time}",activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler("logs/profiles"), schedule=schedule(wait=1, warmup=1, active=3, repeat=2))
        
    log.info(config.name)

    # Init Lightning callbacks and loggers
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")
    
    # Init Lightning trainer
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer,
        profiler=profiler,
        callbacks=callbacks,
        logger=loggers,  # , deterministic=True
        enable_checkpointing=checkpointing,
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    cfg_utils.log_hyperparameters(
        config=config,
        model=emulator_model,
        data_module=data_module,
        trainer=trainer,
        callbacks=callbacks,
    )


    emissionTracker = EmissionsTracker() if emissions_tracker_enabled else None
    if emissionTracker and config.logger.get("wandb"):
        emissionTracker.start()
        
    trainer.fit(model=emulator_model, datamodule=data_module)
    if emissionTracker and config.logger.get("wandb"):
        emissions:float = emissionTracker.stop()
        log.info(f"Total emissions: {emissions} kgCO2")
        cfg_utils.save_emissions_to_wandb(config, emissions)
    
    if(config.logger.get("wandb")):
        cfg_utils.save_hydra_config_to_wandb(config)

    # Testing:
    if(config.logger.get("wandb")):
        if config.get("test_after_training"):
            trainer.test(datamodule=data_module, ckpt_path="best")

        if config.get("logger"):
            wandb.finish()

    # log.info("Reloading model from checkpoint based on best validation stat.")
    # final_model = emulator_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
    #    datamodule_config=config.datamodule, output_normalizer=data_module.normalizer.output_normalizer)
    # return final_model
