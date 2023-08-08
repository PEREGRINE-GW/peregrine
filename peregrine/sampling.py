print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Bilby Sampling
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)


import sys
import numpy as np
from datetime import datetime
import pickle
import swyft.lightning as sl
from config_utils import read_config, init_config
from simulator_utils import init_simulator, simulate
import subprocess
import bilby
from bilby.gw import detector, conversion

bilby.core.utils.logger.setLevel("WARNING")

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [sampling.py] | Reading config file"
    )
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    simulator = init_simulator(conf)
    priors = {}
    for key in conf["priors"]["int_priors"].keys():
        priors[key] = conf["priors"]["int_priors"][key]
    for key in conf["priors"]["ext_priors"].keys():
        priors[key] = conf["priors"]["ext_priors"][key]
    savedir = conf["zarr_params"]["store_path"]
    if conf["tmnre"]["generate_obs"]:
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [sampling.py] | Generating observation"
        )
        obs = simulator.generate_observation()
        with open(f"{savedir}/observation_{conf['zarr_params']['run_id']}", "wb") as f:
            pickle.dump(obs, f)
    else:
        observation_path = conf["tmnre"]["obs_path"]
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [sampling.py] | Loading observation from {observation_path}"
        )
        with open(observation_path, "rb") as f:
            obs = pickle.load(f)
        subprocess.run(
            f"cp {observation_path} {savedir}/observation_{conf['zarr_params']['run_id']}",
            shell=True,
        )

    x0f = obs["d_f"] + obs["n_f"]
    fd_strain = {}
    for idx, ifo in enumerate(conf["waveform_params"]["ifo_list"]):
        fd_strain[ifo] = np.array(x0f[2 * idx, :] + 1j * x0f[2 * idx + 1, :])

    ifos = detector.InterferometerList(conf["waveform_params"]["ifo_list"])
    for ifo in ifos:
        ifo.set_strain_data_from_frequency_domain_strain(
            fd_strain[str(ifo.name)],
            sampling_frequency=conf["waveform_params"]["sampling_frequency"],
            duration=conf["waveform_params"]["duration"],
            start_time=conf["waveform_params"]["start"],
        )

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [sampling.py] | Evaluating likelihood..."
    )
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=simulator.waveform_generator,
        priors=priors,
        jitter_time=conf["sampling"]["time"],
        distance_marginalization=conf["sampling"]["distance"],
        time_marginalization=conf["sampling"]["time"],
        phase_marginalization=conf["sampling"]["phase"],
    )
    start_time = datetime.now()
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [sampling.py] | Starting sampler at {start_time}"
    )
    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        label=conf["sampling"]["sampler"],
        sampler=conf["sampling"]["sampler"],
        outdir=savedir,
        verbose=False,
        **conf["sampler_hparams"][conf["sampling"]["sampler"]],
        conversion_function=conversion.generate_all_bbh_parameters,
        resume=conf["sampling"]["resume_from_ckpt"],
    )
    end_time = datetime.now()
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [sampling.py] | Sampling completed using {conf['sampling']['sampler']} in {end_time - start_time}."
    )
