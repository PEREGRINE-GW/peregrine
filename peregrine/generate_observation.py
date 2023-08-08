print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Generate Observation
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | August 2023
"""
)

import sys
import pickle
from datetime import datetime
from config_utils import read_config, init_config
from simulator_utils import init_simulator

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [generate_observation.py] | Reading config file"
    )
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    simulator = init_simulator(conf)
    obs = simulator.generate_observation()
    with open(
        f"{conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
        "wb",
    ) as f:
        pickle.dump(obs, f)
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [generate_observation.py] | Generated observation"
    )
    print(f"\nConfig: {args[0]}")
    print(
        f"Observation Path: {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
