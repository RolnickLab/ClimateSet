import os
from typing import Optional, Dict, Sequence, Any

import hydra
import torch
from omegaconf import DictConfig
import wandb

from emulator.src.datamodules.dummy_datamodule import DummyDataModule
from emulator.src.utils.utils import get_logger
from emulator.src.utils.wandb_api import (
    load_hydra_config_from_wandb,
    restore_model_from_wandb_cloud,
    get_wandb_ckpt_name,
)
import emulator.src.utils.config_utils as cfg_utils
from emulator.src.core.models.decoder_wrapper import DecoderWrapper

"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""
log = get_logger()


def get_model(config: DictConfig, **kwargs):
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra yaml config file parsing)
    Returns:
        The model that you can directly use to train with pytorch-lightning
    """
    # if config.get('normalizer'):
    # This can be a bit redundant with get_datamodule (normalizer is instantiated twice), but it is better to be
    # sure that the output_normalizer is used by the model in cases where pytorch-lightning is not used.
    # By default if you use pytorch-lightning, the correct output_normalizer is passed to the model before training,
    # even without the below

    # normalizer: Normalizer = hydra.utils.instantiate(
    #    config.normalizer, _recursive_=False,
    #    datamodule_config=config.datamodule,
    # )
    # kwargs['output_normalizer'] = normalizer.output_normalizer

    if config.datamodule.get("name") == "climate_super":
        config.model["super_emulation"] = True  # we load multiple models
    if config.model.get("finetune") is True:
        log.info("Finetuning")
        assert (
            config.model.get("pretrained_run_id") is not None
        ), "Mode is finetune but no run id is given to load from."
        assert (
            config.model.get("pretrained_ckpt_dir") is not None
        ), "Mode is finetune but no run id is given to load from."
        # load pretrained model
        model, _ = reload_model_from_id(
            config.model.get("pretrained_run_id"),
            config.model.get("pretrained_ckpt_dir"),
            allow_resume=False,
        )
        log.warn("Loading pretrained Base model")

        if config.get("decoder") is not None:
            if (config.model.get("pretrained_run_id_decoder") is not None) and (
                config.model.get("pretrained_ckpt_dir_decoder") is not None
            ):
                # reloading decoder
                log.warn("Loading pretrained Decoder")
                model, _ = reload_model_from_id(
                    config.model.get("pretrained_run_id_decoder"),
                    config.model.get("pretrained_ckpt_dir_decoder"),
                    allow_resume=False,
                )

            else:
                log.warn("Creating new Decoder")
                multihead_decoder = hydra.utils.instantiate(config.decoder)
                # only set to True when we have a decoder
                model.hparams["super_decoder"] = True

                model = DecoderWrapper(model, multihead_decoder, **model.hparams)

    # check if we are also finetuning a decoder
    else:
        model = hydra.utils.instantiate(
            config.model,
            _recursive_=False,
            datamodule_config=config.datamodule,
            **kwargs,
        )

        if config.get("decoder") is not None:
            multihead_decoder = hydra.utils.instantiate(config.decoder)
            # only set to True when we have a decoder
            model.hparams["super_decoder"] = True

            model = DecoderWrapper(model, multihead_decoder, **model.hparams)

    return model


def get_datamodule(config: DictConfig) -> DummyDataModule:
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra yaml config file parsing)
    Returns:
        A datamodule that you can directly use to train pytorch-lightning models
    """
    # First we instantiate our normalization preprocesser, then our datamodule, and finally the model
    # normalizer: Normalizer = hydra.utils.instantiate(
    #    config.normalizer,
    #    datamodule_config=config.datamodule,
    #    _recursive_=False
    # )

    data_module: DummyDataModule = hydra.utils.instantiate(
        config.datamodule,
        # input_transform=config.model.get("input_transform"),
        # normalizer=normalizer
    )

    return data_module


def get_model_and_data(config: DictConfig):
    """
    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra yaml config file parsing)
    Returns:
        A tuple of (model, datamodule), that you can directly use to train with pytorch-lightning
        (e.g., checkpointing the best model w.r.t. a small validation set with the ModelCheckpoint callback),
        with:
            trainer.fit(model=model, datamodule=datamodule)
    """
    data_module = get_datamodule(config)
    model = get_model(config)

    return model, data_module


def reload_model_from_config_and_ckpt(
    config: DictConfig, model_path: str, load_datamodule: bool = True
):
    model, data_module = get_model_and_data(config)
    # Reload model
    model_state = torch.load(model_path)["state_dict"]
    model.load_state_dict(model_state)
    if load_datamodule:
        return model, data_module
    return model


def reload_model_from_config_and_ckpt(
    config: DictConfig,
    model_path: str,
    device: Optional[torch.device] = None,
    load_datamodule: bool = False,
) -> Dict[str, Any]:
    """Load a model as defined by ``config.model`` and reload its weights from ``model_path``.


    Args:
        config: The config to use to reload the model
        model_path: The path to the model checkpoint (its weights)
        device: The device to load the model on. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        BaseModel: The reloaded model if load_datamodule is ``False``, otherwise a tuple of (reloaded-model, datamodule)

    Examples:

    .. code-block:: python

        # If you used wandb to save the model, you can use the following to reload it
        from climart.utilities.wandb_api import load_hydra_config_from_wandb

        run_path = group/PROJECT/RUN_ID   # wandb run id (you can find it on the wandb URL after runs/, e.g. 1f5ehvll)
        config = load_hydra_config_from_wandb(run_path, override_kwargs=['datamodule.num_workers=4', 'trainer.gpus=-1'])

        model, datamodule = reload_model_from_config_and_ckpt(config, model_path, load_datamodule=True)

        # Test the reloaded model
        trainer = hydra.utils.instantiate(config.trainer, _recursive_=False)
        trainer.test(model=model, datamodule=datamodule)

    """
    # First we instantiate our normalization preprocesser, then our datamodule, and finally the model

    # normalizer: Normalizer = hydra.utils.instantiate(
    #    config.normalizer, datamodule_config=config.datamodule, _recursive_=False
    # )
    normalizer = None  # TODO: we might want to ahve normalizers

    if load_datamodule:
        data_module = hydra.utils.instantiate(
            config.datamodule,
            input_transform=config.model.get("input_transform"),
            normalizer=normalizer,
        )
    else:
        data_module = None

    # model, data_module = get_model_and_data(config)
    model = get_model(config)
    # Reload model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(model_path, map_location=device)
    # Reload weights
    model.load_state_dict(model_state["state_dict"])
    epoch, global_step = model_state["epoch"], model_state["global_step"]
    return {
        "model": model,
        "datamodule": data_module,
        "epoch": epoch,
        "global_step": global_step,
    }


def reload_model_from_id(
    run_id: str,
    direc: str = None,
    group="causalpaca",
    project="emulator",
    override_kwargs: Sequence[str] = None,
    allow_resume: bool = True,
):
    """
    This function reloads a model from a wandb run id
        -> If test_model is True, it tests the model (i.e. can be used to test a trained model).
    If the model was trained using Wandb logging, reloading it and resuming training or testing will be as easy as:
        >>> example_run_id = "22ejv03e"

    Args:
        run_id: Wandb run id
        checkpoint_path: An optional local ckpt path to load the weights from. If None, the best one on wandb will be used.
        config: An optional config to load the model and data from. If None, the config is loaded from Wandb.
        group: Wandb group
        project: Wandb project
        override_kwargs: A list of strings (of the form "key=value") to override the given/reloaded config with.
        allow_resume: Wheather resuming of training is allowed or a new instance should be created.

    """
    run_path = f"{group}/{project}/{run_id}"

    if os.path.isdir(os.path.join(direc, run_id)):
        saved_ckpts = [
            f for f in os.listdir(os.path.join(direc, run_id)) if f.endswith(".ckpt")
        ]
        log.info(" Checkpoints saved:", saved_ckpts)
        if "last.ckpt" in saved_ckpts:
            log.info("Reloading from last.ckpt")
            checkpoint_path = os.path.join(direc, f"{run_id}/last.ckpt")
        else:
            log.info("Reloading from", saved_ckpts[0])
            checkpoint_path = os.path.join(direc, f"{run_id}/{saved_ckpts[0]}")
    else:
        checkpoint_path = direc

    if checkpoint_path is not None:  # local loading
        print("Loading checkpoint from local.")
        if checkpoint_path.endswith(".ckpt"):
            best_model_path = checkpoint_path
        else:
            try:
                best_model_path = checkpoint_path + "/" + get_wandb_ckpt_name(run_path)
            except IndexError:
                saved_files = [f.name for f in wandb.Api().run(run_path).files()]
                log.warning(
                    f"Run {run_id} does not have a saved ckpt in {checkpoint_path}. All saved files: {saved_files}"
                )
                best_model_path = restore_model_from_wandb_cloud(run_path)
    else:
        best_model_path = restore_model_from_wandb_cloud(run_path)

    config = load_hydra_config_from_wandb(run_path, override_kwargs)

    if not (allow_resume):
        config.logger["wandb"]["resume"] = False
        config.logger["wandb"]["reinit"] = False

    if allow_resume:
        cfg_utils.extras(config)

    reloaded = reload_model_from_config_and_ckpt(
        config, best_model_path, load_datamodule=False
    )
    model = reloaded["model"]

    print("Got reloaded model.")

    return model, config.datamodule


if __name__ == "__main__":
    print("hello")
