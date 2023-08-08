print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Load Simulator
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
import swyft.lightning as sl
from datetime import datetime
from config_utils import read_config, init_config
from simulator_utils import init_simulator

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        "{datetime.now().strftime('%a %d %b %H:%M:%S')} | [load_simulator.py] | Reading config file"
    )
    # Load and parse config file
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    simulator = init_simulator(conf)
