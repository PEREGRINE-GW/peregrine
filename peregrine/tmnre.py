print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: TMNRE Inference
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
from datetime import datetime
import glob
import pickle
import logging
from pathlib import Path

import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from sbi.inference import SNPE, SNPE_A  # Replace swyft with sbi
from sbi.inference.posteriors import DirectPosterior
from sbi.utils.torchutils import process_device
from config_utils import read_config, init_config
from simulator_utils import init_simulator, Simulator  # Keep simulator initialization
from inference_utils import (
    save_bounds,
    load_bounds,
    save_posterior,  # New function to save posterior
    load_posterior, # New function to load posterior
)
import subprocess

# For parallelisation
import psutil


def setup_inference(conf: dict, prior, round_id: int):
    """Set up the SBI inference object.

    Args:
        conf: Configuration dictionary.
        prior: Prior distribution.
        round_id: Current round.

    Returns:
        inference: SBI inference object.
        trainer_dir: Directory for training logs.
    """
    trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R{round_id}"

    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    
    if conf["tmnre"]["density_estimator"] == "SNPE_A":
        inference = SNPE_A(prior=prior, device=conf["device_params"]["device"])
    else:
        inference = SNPE(prior=prior, device=conf["device_params"]["device"])
    
    return inference, trainer_dir


def train_posterior(
    inference,
    train_data,
    val_data,
    trainer_dir: str,
    conf: dict,
    round_id: int,
    retrain_from_scratch = False,
):
    """Train the posterior estimator.

    Args:
        inference: SBI inference object.
        train_data: Training dataloader.
        val_data: Validation dataloader.
        trainer_dir: Directory for training logs.
        conf: Configuration dictionary.
        round_id: Current round.
        retrain_from_scratch: Whether to retrain the posterior from scratch.
    
    Returns:
        posterior: Trained posterior estimator.
    """
    if not conf["tmnre"]["infer_only"] or not Path(f"{trainer_dir}/posterior_R{round_id}.pkl").exists() or retrain_from_scratch:
        # Train inference
        start_time = datetime.now()
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Training network for round {round_id} at {start_time}"
        )

        theta, x, _ = train_data.dataset.tensors  # Access tensors from the dataset

        inference.append_simulations(theta.to(conf["device_params"]["device"]), x.to(conf["device_params"]["device"]))

        density_estimator = inference.train(
            training_batch_size=conf["hparams"]["training_batch_size"],
            learning_rate=conf["hparams"]["learning_rate"],
            validation_fraction=0.1,
            stop_after_epochs=conf["hparams"]["early_stopping"],
            max_num_epochs=conf["hparams"]["max_epochs"],
        )

        posterior = inference.build_posterior(density_estimator, sample_with="mcmc")

        end_time = datetime.now()
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Training completed for round {round_id} in {end_time - start_time}."
        )
        save_posterior(posterior, conf, round_id) # Save the posterior
        logging.info(f"Posterior saved for round {round_id}")
    else:
        posterior = load_posterior(conf, round_id)

    return posterior
