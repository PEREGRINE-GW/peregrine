print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Coverage Test
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
import numpy as np
from datetime import datetime
import glob
import swyft.lightning as sl
from config_utils import read_config, init_config
from simulator_utils import init_simulator
from inference_utils import (
    save_coverage,
    setup_zarr_store,
    setup_dataloader,
    setup_trainer,
    init_network,
    load_bounds,
)
import subprocess
import psutil
import time


if __name__ == "__main__":
    args = sys.argv[1:]
    n_samples = int(args[1])
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Running coverage tests on {n_samples} samples per round"
    )
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Reading config file"
    )
    # Load and parse config file
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    conf["tmnre"]["shuffling"] = False
    round_id = int(conf["tmnre"]["num_rounds"])
    bounds = load_bounds(conf, round_id)
    simulator = init_simulator(conf, bounds)
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Initialising coverage zarrstore for round {round_id}"
    )
    coverage_store = setup_zarr_store(
        conf, simulator, round_id=round_id, coverage=True, n_sims=n_samples
    )
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Simulating coverage observations for round {round_id}"
    )
    if conf["zarr_params"]["njobs"] == -1:
        njobs = psutil.cpu_count(logical=True)
    elif conf["zarr_params"]["njobs"] > psutil.cpu_count(logical=False):
        njobs = psutil.cpu_count(logical=True)
    elif conf["zarr_params"]["run_parallel"]:
        njobs = conf["zarr_params"]["njobs"]
    else:
        njobs = 1
    while coverage_store.sims_required > 0:
        processes = []
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | (Re)starting {njobs} simulation batches. {coverage_store.sims_required} simulations still required"
        )
        for job in range(njobs):
            p = subprocess.Popen(
                [
                    "python",
                    "run_parallel.py",
                    f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.txt",
                    str(round_id),
                    f"coverage",
                ]
            )
            processes.append(p)
        status_array = np.array([None for p in processes])
        while np.all(status_array == None) and len(status_array) != 0:
            time.sleep(60)
            status_array = np.array([p.poll() for p in processes])
        for p in processes:
            p.kill()

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Setting up dataloaders for round {round_id}"
    )
    train_data, val_data, trainer_dir = setup_dataloader(
        coverage_store, simulator, conf, round_id
    )

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Setting up trainer for round {round_id}"
    )
    trainer = setup_trainer(trainer_dir, conf, round_id)

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Initialising network for round {round_id}"
    )
    network = init_network(conf)

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Generate prior samples"
    )
    prior_samples = simulator.sample(100_000, targets=["z_total"])

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Generate posterior samples"
    )
    trainer.test(
        network, val_data, glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0]
    )
    coverage_samples = trainer.test_coverage(network, coverage_store[0:], prior_samples)
    save_coverage(coverage_samples, conf, round_id)

    sys.exit(0)
