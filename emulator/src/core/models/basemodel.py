import hydra

import numpy as np

import time

from typing import Optional, List, Any, Dict, Tuple, Union

from omegaconf import DictConfig
import omegaconf
from pytorch_lightning import LightningModule
import torch

from emulator.src.core.evaluation import evaluate_preds, evaluate_per_target_variable
from emulator.src.utils.utils import get_loss_function, get_logger, to_DictConfig

# from emulator.src.utils.interface import reload_model_from_id
from emulator.src.core.callbacks import PredictionPostProcessCallback
from timm.optim import create_optimizer_v2


class BaseModel(LightningModule):
    """Abstract template class for all NN based emulators.
    Each model that inherits from BaseModel must implement the __init__ and
    forward method. Functions provided here can be overriden if needed!
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    # TODO normalization / transformation configs and more
    def __init__(
        self,
        datamodule_config: DictConfig = None,
        optimizer: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        name: str = "",
        verbose: bool = True,
        loss_function: str = "mean_squared_error",
        monitor: Optional[str] = None,
        mode: str = "min",
        finetune: bool = False,
        super_emulation: bool = False,
        super_decoder: bool = False,
        pretrained_run_id: Optional[Union[str, None]] = None,
        pretrained_ckpt_dir: Optional[Union[str, None]] = None,
        **kwargs,
    ):
        super().__init__()

        self.log_text = get_logger(__name__)
        self.log_text.info("Base Model init!")

        self.name = name
        self.verbose = verbose
        self.test_step_outputs = {}

        self.val_step_outputs = []

        self.criterion = get_loss_function(loss_function)
        self.super_emulation = super_emulation
        self.log_text.info(f"Super Emulation: {self.super_emulation}")
        self.super_decoder = super_decoder
        self.log_text.info(f"Super Decoder: {self.super_decoder}")

        if datamodule_config is not None:
            # get information from data config

            self._out_var_ids = datamodule_config.get("out_var_ids")
            self.num_levels = datamodule_config.get("num_levels")
            self.output_postprocesser = PredictionPostProcessCallback(
                variables=self._out_var_ids, sizes=self.num_levels
            )

        if not hasattr(self.hparams, "monitor") or self.hparams.monitor is None:
            self.hparams.monitor = f"val/llrmse_climax"
        if not hasattr(self.hparams, "mode") or self.hparams.mode is None:
            self.hparams.mode = "min"

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _apply(self, fn):
        super(BaseModel, self)._apply(fn)
        # if self.output_normalizer is not None:
        #    self.output_normalizer.apply_torch_func(fn)
        return self

    def forward(self, X):
        """
        Downstream model forward pass, input X will be the (batched) output from self.input_transform
        """
        raise NotImplementedError("Base model is an abstract class!")

    def on_train_start(self) -> None:
        self.log_text.info("Starting Training")

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()

    def predict(self, X, idx, *args, **kwargs):
        # x (batch_size, time, lon, lat, num_features)
        # TODO if we want to apply any input normalization or other stuff we should do it here

        # if idx is None or if we do not have a decoder

        if self.super_decoder:
            assert idx is not None, "Super Decoder but model index is None"
            preds = self(X, idx)
        else:
            preds = self(X)

        # TODO if we want to apply any output normalization we should do it here
        # else we will just return raw predictions

        # splitting predictions to get dict accessible via target var id
        preds_dict = self.output_postprocesser.split_vector_by_variable(preds)

        return preds_dict

    def training_step(self, batch: Any, batch_idx: int):
        if self.super_emulation:
            X, Y, idx = batch
        else:
            X, Y = batch
            idx = None

        preds = self.predict(X, idx)
        # dict with keys being the output var ids
        Y = self.output_postprocesser.split_vector_by_variable(
            Y
        )  # split per var id #TODO: might need to remove that for other datamodule

        train_log = dict()  # everything we want to log to wandb should go in here

        loss = 0

        #  Loop over output variable to compute loss seperateley!!!
        for out_var in self._out_var_ids:
            loss_per_var = self.criterion(preds[out_var], Y[out_var])
            if torch.isnan(loss_per_var).sum() > 0:
                exit(0)
            loss += loss_per_var
            train_log[f"train/{out_var}/loss"] = loss_per_var
            # any additional losses can be computed, logged and added to the loss here

        # Average Loss over vars
        loss = loss / len(self._out_var_ids)

        n_zero_gradients = (
            sum(
                [
                    int(torch.count_nonzero(p.grad == 0))
                    for p in self.parameters()
                    if p.grad is not None
                ]
            )
            / self.n_params
        )

        self.log_dict(
            {**train_log, "train/loss": loss, "n_zero_gradients": n_zero_gradients}
        )

        ret = {
            "loss": loss,
            "n_z                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ero_gradients": n_zero_gradients,
            "targets": Y,
            "preds": preds,
        }

        return ret

    def on_train_epoch_end(self):
        train_time = time.time() - self._start_epoch_time
        self.log_dict({"epoch": self.current_epoch, "time/train": train_time})

    def _evaluation_step(self, batch: Any, batch_idx: int):
        if self.super_emulation:
            X, Y, idx = batch
        else:
            X, Y = batch
            idx = None

        preds = self.predict(X, idx)
        ret = {"targets": Y, "preds": preds}

        self.val_step_outputs.append(ret)

        return ret

    def _evaluation_get_preds(
        self, outputs: List[Any]
    ) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        for batch in outputs:
            batch["targets"] = self.output_postprocesser.split_vector_by_variable(
                batch["targets"]
            )  # TODO: we might want to remove that for the real data module

        Y = {
            out_var: torch.cat([batch["targets"][out_var] for batch in outputs], dim=0)
            .cpu()
            .numpy()
            for out_var in self._out_var_ids
        }
        preds = {
            out_var: torch.cat([batch["preds"][out_var] for batch in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
            for out_var in self._out_var_ids
        }

        # any additional information from input we want should be extracted here
        # TODO

        return {"targets": Y, "preds": preds}

    def on_validation_epoch_start(self):
        self._start_validation_epoch_time = time.time()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> dict:
        val_time = time.time() - self._start_validation_epoch_time
        self.log("time/validation", val_time)

        validation_outputs = self._evaluation_get_preds(self.val_step_outputs)
        # get Ytrue and preds
        Ytrue, preds = validation_outputs["targets"], validation_outputs["preds"]

        val_stats = evaluate_per_target_variable(Ytrue, preds, data_split="val")
        target_val_metric = val_stats.pop(self.hparams.monitor)
        self.log_dict({**val_stats, "epoch": self.current_epoch}, prog_bar=False)

        # Show the main validation metric on the progress bar:
        self.log(self.hparams.monitor, target_val_metric, prog_bar=True)
        self.val_step_outputs.clear()  # clear memory

        return val_stats

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = time.time()
        for i, _ in enumerate(self.trainer.datamodule.test_set_names):
            self.test_step_outputs[i] = []  # initialize output collection

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        ret = self._evaluation_step(batch, batch_idx)
        self.test_step_outputs[dataloader_idx].append(ret)  # collect for evaluation
        return ret

    def on_test_epoch_end(self) -> dict:
        test_time = time.time() - self._start_test_epoch_time
        self.log("time/test", test_time)

        outputs = (
            self.test_step_outputs
        )  # we ned to collect and clear our outpets as ourselves (new in python 10)
        main_test_stats = dict()

        self.log_text.info(f"in test epoch end len outputs {len(outputs)}")

        # test statistisc per test set
        for i, split_name in enumerate(self.trainer.datamodule.test_set_names):
            self.log_text.info(f"Testing on {split_name}")
            test_subset_outputs = self._evaluation_get_preds(outputs[i])

            Y_test, preds_test = (
                test_subset_outputs["targets"],
                test_subset_outputs["preds"],
            )

            split_name = f"test/{split_name}"

            test_stats = evaluate_per_target_variable(
                Y_test, preds_test, data_split=split_name
            )

            self.log_dict({**test_stats, "epoch": self.current_epoch}, prog_bar=False)

        self.log_dict({**main_test_stats, "epoch": self.current_epoch}, prog_bar=False)
        self.test_step_outputs.clear()
        return main_test_stats

    def aggregate_predictions(
        self, results: List[Any]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Args:
         results: The list that pl.Trainer() returns when predicting, i.e.
                        results = trainer.predict(model, datamodule)
        Returns:
            A dict mapping prediction_set_name_i -> {'targets': t_i, 'preds': p_i}
                for each prediction subset i .
                E.g.: To access the shortwave heating rate predictions for year 2012:
                    model, datamodule, trainer = ...
                    datamodule.predict_years = "2012"
                    results = trainer.predict(model, datamodule)
                    results = model.aggregate_predictions(results)
                    sw_hr_preds_2012 = results[2012]['preds']['hrsc']
        """
        if not isinstance(results[0], list):
            results = [results]  # when only a single predict dataloader is passed
        per_subset_outputs = dict()
        for pred_set_name, predict_subset_outputs in zip(
            self.trainer.datamodule.predict_years, results
        ):
            Y, preds = self._evaluation_get_preds(predict_subset_outputs)
            per_subset_outputs[pred_set_name]["preds"] = preds
            per_subset_outputs[pred_set_name]["targets"] = Y
        return per_subset_outputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    def on_predict_epoch_end(
        self, results: List[Any]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        return self.aggregate_predictions(results)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    def configure_optimizers(self):
        # configuring optimizer and lr schedul from dict configs

        if "_target_" in to_DictConfig(self.hparams.optimizer).keys():
            self.hparams.optimizer.name = str(
                self.hparams.optimizer._target_.split(".")[-1]
            ).lower()
        if "name" not in to_DictConfig(self.hparams.optimizer).keys():
            self.log_text.info(
                " No optimizer was specified, defaulting to AdamW with 1e-4 lr."
            )
            self.hparams.optimizer.name = "adamw"

        if hasattr(self, "no_weight_decay"):
            self.log_text.info(f"Model has method no_weight_decay, which will be used.")
        optim_kwargs = {
            k: v
            for k, v in self.hparams.optimizer.items()
            if k not in ["name", "is_filtered", "_target_"]
        }
        try:
            if self.hparams.optimizer.is_filtered is not None and self.hparams.optimizer.is_filtered:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.parameters()), **optim_kwargs
                )
            else:
                optimizer = create_optimizer_v2(
                    model_or_params=self, opt=self.hparams.optimizer.name, **optim_kwargs
                )
        except omegaconf.errors.ConfigAttributeError:
            # attribute does not exist (reloading old models)
            optimizer = create_optimizer_v2(
                    model_or_params=self, opt=self.hparams.optimizer.name, **optim_kwargs
                )

        self._init_lr = optimizer.param_groups[0]["lr"]

        if self.hparams.scheduler is None:
            return optimizer  # no scheduler
        else:
            if "_target_" not in to_DictConfig(self.hparams.scheduler).keys():
                raise ValueError(
                    "Please provide a _target_ class for model.scheduler arg!"
                )
            scheduler_params = to_DictConfig(self.hparams.scheduler)
            scheduler = hydra.utils.instantiate(scheduler_params, optimizer=optimizer)

        #  what metric to monitor for training
        if not hasattr(self.hparams, "monitor") or self.hparams.monitor is None:
            self.hparams.monitor = f"val/rmse"
        if not hasattr(self.hparams, "mode") or self.hparams.mode is None:
            self.hparams.mode = "min"

        lr_dict = {
            "scheduler": scheduler,
            "monitor": self.hparams.monitor,
            "mode": self.hparams.mode,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}


if __name__ == "__main__":
    pretrained_run_id = "3usp9c7m"
    pretrained_ckpt_dir = "emulator/emulator/"
    base_model, _ = reload_model_from_id(
        pretrained_run_id, pretrained_ckpt_dir, allow_resume=False
    )
    print(base_model)
