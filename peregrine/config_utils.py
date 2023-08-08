import configparser
import subprocess
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import bilby
from datetime import datetime

bilby.core.utils.logger.setLevel("WARNING")
from bilby.gw import prior as prior_gw
from bilby.core import prior as prior_core
from bilby.gw import source, conversion


def read_config(sysargs: list):
    """
    Load the config using configparser, returns a parser object that can be accessed as
    e.g. tmnre_parser['FIELD']['parameter'], this will always return a string, so must be
    parsed for data types separately, see init_config
    Args:
      sysargs: list of command line arguments (i.e. strings) containing path to config in position 0
    Returns:
      Config parser object containing information in configuration file sections
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> tmnre_parser['TMNRE']['num_rounds'] etc.
    """
    tmnre_config_file = sysargs[0]
    tmnre_parser = configparser.ConfigParser()
    tmnre_parser.read_file(open(tmnre_config_file))
    return tmnre_parser


def init_config(tmnre_parser, sysargs: list, sim: bool = False) -> dict:
    """
    Initialise the config dictionary, this is a dictionary of dictionaries obtaining by parsing
    the relevant config file. A copy of the config file is stored along with the eventual simulations.
    All parameters are parsed to the correct data type from strings, including lists and booleans etc.
    Args:
      tmnre_parser: config parser object, output of read_config
      sysargs: list of command line arguments (i.e. strings) containing path to config in position 0
      sim: boolean to choose whether to include config copying features etc. if False, will create a
           copy of the config in the store directory and generate the param idxs file
    Returns:
      Dictionary of configuration options with all data types explicitly parsed
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, )
    """
    tmnre_config_file = sysargs[0]
    store_path = tmnre_parser["ZARR PARAMS"]["store_path"]
    if not sim:
        # make sure path exists
        Path(store_path).mkdir(parents=True, exist_ok=True)
    run_id = tmnre_parser["ZARR PARAMS"]["run_id"]
    if not sim:
        subprocess.run(
            f"cp {tmnre_config_file} {store_path}/config_{run_id}.txt", shell=True
        )
    conf = {}

    gw_source = {}
    for key in tmnre_parser["SOURCE"]:
        if key in ["source_type"]:
            gw_source[key] = str(tmnre_parser["SOURCE"][key])
        elif key in ["aligned_spins"]:
            gw_source[key] = bool(strtobool(tmnre_parser["SOURCE"][key]))
    if gw_source["source_type"] in ["BBH"]:
        gw_source["fd_source_model"] = source.lal_binary_black_hole
        gw_source[
            "param_conversion_model"
        ] = conversion.convert_to_lal_binary_black_hole_parameters
    elif gw_source["source_type"] in ["NSBH", "BNS"]:
        gw_source["fd_source_model"] = source.lal_binary_neutron_star
        gw_source[
            "param_conversion_model"
        ] = conversion.convert_to_lal_binary_neutron_star_parameters
    conf["source"] = gw_source

    waveform_params = {}
    for key in tmnre_parser["WAVEFORM PARAMS"]:
        if key in [
            "sampling_frequency",
            "minimum_frequency",
            "maximum_frequency",
            "reference_frequency",
            "start_frequency_bin",
            "end_frequency_bin",
        ]:
            waveform_params[key] = int(tmnre_parser["WAVEFORM PARAMS"][key])
        elif key in ["duration", "start_offset", "start"]:
            waveform_params[key] = float(tmnre_parser["WAVEFORM PARAMS"][key])
        elif key in ["waveform_apprx"]:
            waveform_params[key] = str(tmnre_parser["WAVEFORM PARAMS"][key])
        elif key in ["ifo_list"]:
            waveform_params[key] = [
                str(ifo) for ifo in tmnre_parser["WAVEFORM PARAMS"][key].split(",")
            ]
        elif key in ["ifo_noise"]:
            waveform_params[key] = bool(strtobool(tmnre_parser["WAVEFORM PARAMS"][key]))
    conf["waveform_params"] = waveform_params

    injection = {}
    for key in tmnre_parser["INJECTION"]:
        if key in [None]:
            continue
        else:
            injection[key] = np.float64(tmnre_parser["INJECTION"][key])
    # Keep either mass_1, mass_2 or mass_ratio, chirp_mass
    if "mass_1" in tmnre_parser["INT PRIORS"].keys():
        ip = injection.copy()
        for key in ["mass_ratio", "chirp_mass"]:
            ip.pop(key)
    else:
        ip = injection.copy()
        for key in ["mass_1", "mass_2"]:
            ip.pop(key)
    conf["injection"] = ip

    int_priors, ext_priors = populate_priors(tmnre_parser)
    priors = {"int_priors": int_priors, "ext_priors": ext_priors}
    conf["priors"] = priors
    fixed = []
    varying = []
    for key in ip.keys():
        if key in int_priors.keys() or key in ext_priors.keys():
            varying.append(key)
        else:
            fixed.append(key)
    conf["fixed"] = fixed
    conf["varying"] = varying

    param_idxs = {}
    param_names = {}
    if not sim:
        with open(
            (
                f"{tmnre_parser['ZARR PARAMS']['store_path']}/param_idxs_{tmnre_parser['ZARR PARAMS']['run_id']}.txt"
            ),
            "w",
        ) as f:
            for idx, key in enumerate(int_priors.keys()):
                param_idxs[key] = idx
                param_names[idx] = key
                if key in varying:
                    f.write(f"{key} {idx} intrinsic varying\n")
                else:
                    f.write(f"{key} {idx} intrinsic fixed\n")
            for idx, key in enumerate(ext_priors.keys()):
                param_idxs[key] = idx + len(int_priors.keys())
                param_names[idx + len(int_priors.keys())] = key
                if key in varying:
                    f.write(f"{key} {idx + len(int_priors.keys())} extrinsic varying\n")
                else:
                    f.write(f"{key} {idx + len(int_priors.keys())} extrinsic fixed\n")
            f.close()
    else:
        for idx, key in enumerate(int_priors.keys()):
            param_idxs[key] = idx
            param_names[idx] = key
        for idx, key in enumerate(ext_priors.keys()):
            param_idxs[key] = idx + len(int_priors.keys())
            param_names[idx + len(int_priors.keys())] = key
    conf["param_idxs"] = param_idxs
    conf["vary_idxs"] = [param_idxs[key] for key in varying]
    conf["param_names"] = param_names

    tmnre = {}
    tmnre["infer_only"] = False
    tmnre["marginals"] = None
    for key in tmnre_parser["TMNRE"]:
        if key in ["num_rounds"]:
            tmnre[key] = int(tmnre_parser["TMNRE"][key])
        elif key in ["bounds_th"]:
            tmnre[key] = float(tmnre_parser["TMNRE"][key])
        elif key in ["1d_only", "generate_obs", "resampler", "infer_only", "shuffling"]:
            tmnre[key] = bool(strtobool(tmnre_parser["TMNRE"][key]))
        elif key in ["obs_path"]:
            tmnre[key] = str(tmnre_parser["TMNRE"][key])
        elif key in ["noise_targets"]:
            tmnre[key] = [
                str(target) for target in tmnre_parser["TMNRE"][key].split(",")
            ]
        elif key in ["marginals"]:
            marginals_string = tmnre_parser["TMNRE"][key]
            marginals_list = []
            for marginal in marginals_string.split("("):
                split_marginal = marginal.split(")")
                if split_marginal[0] != "":
                    indices = split_marginal[0].split(",")
                    mg = []
                    for index in indices:
                        mg.append(int(index.strip(" ")))
                    marginals_list.append(tuple(mg))
            tmnre["marginals"] = tuple(marginals_list)
    conf["tmnre"] = tmnre

    if "SAMPLING" in list(tmnre_parser.keys()):
        sampling = {}
        for key in tmnre_parser["SAMPLING"]:
            if key in ["sampler"]:
                sampling[key] = str(tmnre_parser["SAMPLING"][key])
            elif key in [
                "npoints",
                "nsamples",
                "printdt",
                "walks",
                "nact",
                "ntemps",
                "nlive",
                "nwalkers",
            ]:
                sampling[key] = int(tmnre_parser["SAMPLING"][key])
            elif key in ["distance", "time", "phase", "resume_from_ckpt"]:
                sampling[key] = bool(strtobool(tmnre_parser["SAMPLING"][key]))
        sampler_hparams = dict(
            bilby_mcmc=dict(
                n_samples=sampling["nsamples"],
                n_temps=sampling["ntemps"],
                printdt=sampling["printdt"],
            ),
            dynesty=dict(
                nlive=sampling["nlive"], walks=sampling["walks"], nact=sampling["nact"]
            ),
            cpnest=dict(
                nlive=sampling["nlive"], walks=sampling["walks"], nact=sampling["nact"]
            ),
            pymultinest=dict(npoints=sampling["npoints"]),
            ptemcee=dict(
                ntemps=sampling["ntemps"],
                nwalkers=sampling["nwalkers"],
                nsamples=sampling["nsamples"],
            ),
        )
        conf["sampling"] = sampling
        conf["sampler_hparams"] = sampler_hparams

    zarr_params = {}
    for key in tmnre_parser["ZARR PARAMS"]:
        if key in ["run_id", "store_path"]:
            zarr_params[key] = tmnre_parser["ZARR PARAMS"][key]
        elif key in ["use_zarr", "run_parallel"]:
            zarr_params[key] = bool(strtobool(tmnre_parser["ZARR PARAMS"][key]))
        elif key in ["nsims", "chunk_size", "njobs"]:
            zarr_params[key] = int(tmnre_parser["ZARR PARAMS"][key])
        elif key in ["targets"]:
            zarr_params[key] = [
                target
                for target in tmnre_parser["ZARR PARAMS"][key].split(",")
                if target != ""
            ]
        elif key in ["sim_schedule"]:
            zarr_params[key] = [
                int(nsims) for nsims in tmnre_parser["ZARR PARAMS"][key].split(",")
            ]
    if "sim_schedule" in zarr_params.keys():
        if len(zarr_params["sim_schedule"]) != tmnre["num_rounds"]:
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [config_utils.py] | WARNING: Error in sim scheduler, setting to default n_sims = 30_000"
            )
            zarr_params["nsims"] = 30_000
        elif "nsims" in zarr_params.keys():
            zarr_params.pop("nsims")
    conf["zarr_params"] = zarr_params

    hparams = {}
    for key in tmnre_parser["HYPERPARAMS"].keys():
        if key in [
            "min_epochs",
            "max_epochs",
            "early_stopping",
            "num_workers",
            "training_batch_size",
            "validation_batch_size",
        ]:
            hparams[key] = int(tmnre_parser["HYPERPARAMS"][key])
        elif key in ["learning_rate", "train_data", "val_data"]:
            hparams[key] = float(tmnre_parser["HYPERPARAMS"][key])
    conf["hparams"] = hparams

    device_params = {}
    for key in tmnre_parser["DEVICE PARAMS"]:
        if key in ["device"]:
            device_params[key] = str(tmnre_parser["DEVICE PARAMS"][key])
        elif key in ["n_devices"]:
            device_params[key] = int(tmnre_parser["DEVICE PARAMS"][key])
    conf["device_params"] = device_params

    return conf


def populate_priors(tmnre_parser):
    """
    Construct the prior dictionary taking into account the relevant bilby defaults
    Args:
      tmnre_parser: config parser object, output of read_config
    Returns:
      Prior dictionary
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> priors = populate_priors(tmnre_parser)
    """
    # Initialise prior dictionaries
    int_priors = {}
    ext_priors = {}
    # Populate intrinsic and extrinsic prior dictionaries
    for prior_type, prior_dict in zip(["INT", "EXT"], [int_priors, ext_priors]):
        if tmnre_parser["SOURCE"]["source_type"] == "BBH":
            for key in tmnre_parser[f"{prior_type} PRIORS"].keys():
                # Need to treat geocentric time differently
                if key not in ["geocent_time"]:
                    if tmnre_parser["SOURCE"]["aligned_spins"] == "True":
                        prior_dict[key] = prior_gw.BBHPriorDict(aligned_spin=True)[key]
                    elif tmnre_parser["SOURCE"]["aligned_spins"] == "False":
                        prior_dict[key] = prior_gw.BBHPriorDict()[key]
                    if key in tmnre_parser[f"{prior_type} DISTRIBUTIONS"].keys():
                        if tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                            1
                        ] in ["periodic", "reflective"]:
                            prior_dict[key] = getattr(
                                prior_core,
                                tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(
                                    ","
                                )[0],
                            )(
                                minimum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[0]
                                ),
                                maximum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[1]
                                ),
                                boundary=tmnre_parser[f"{prior_type} DISTRIBUTIONS"][
                                    key
                                ].split(",")[1],
                            )
                        else:
                            prior_dict[key] = getattr(
                                prior_core,
                                tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(
                                    ","
                                )[0],
                            )(
                                minimum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[0]
                                ),
                                maximum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[1]
                                ),
                                boundary=None,
                            )
                elif key in ["geocent_time"]:
                    if tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                        1
                    ] in ["periodic", "reflective"]:
                        ext_priors[key] = getattr(
                            prior_core,
                            tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                                0
                            ],
                        )(
                            minimum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[0]
                            ),
                            maximum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[1]
                            ),
                            boundary=tmnre_parser[f"{prior_type} DISTRIBUTIONS"][
                                key
                            ].split(",")[1],
                        )
                    else:
                        ext_priors[key] = getattr(
                            prior_core,
                            tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                                0
                            ],
                        )(
                            minimum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[0]
                            ),
                            maximum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[1]
                            ),
                            boundary=None,
                        )
        elif tmnre_parser["SOURCE"]["source_type"] in ["NSBH", "BNS"]:
            for key in tmnre_parser[f"{prior_type} PRIORS"].keys():
                if key not in ["geocent_time"]:
                    if tmnre_parser["SOURCE"]["aligned_spins"] == "True":
                        prior_dict[key] = prior_gw.BNSPriorDict(aligned_spin=True)[key]
                    elif tmnre_parser["SOURCE"]["aligned_spins"] == "False":
                        prior_dict[key] = prior_gw.BNSPriorDict()[key]
                    if key in tmnre_parser[f"{prior_type} DISTRIBUTIONS"].keys():
                        if tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                            1
                        ] in ["periodic", "reflective"]:
                            prior_dict[key] = getattr(
                                prior_core,
                                tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(
                                    ","
                                )[0],
                            )(
                                minimum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[0]
                                ),
                                maximum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[1]
                                ),
                                boundary=tmnre_parser[f"{prior_type} DISTRIBUTIONS"][
                                    key
                                ].split(",")[1],
                            )
                        else:
                            prior_dict[key] = getattr(
                                prior_core,
                                tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(
                                    ","
                                )[0],
                            )(
                                minimum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[0]
                                ),
                                maximum=float(
                                    tmnre_parser[f"{prior_type} PRIORS"][key].split(
                                        ","
                                    )[1]
                                ),
                                boundary=None,
                            )
                elif key in ["geocent_time"]:
                    if tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                        1
                    ] in ["periodic", "reflective"]:
                        ext_priors[key] = getattr(
                            prior_core,
                            tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                                0
                            ],
                        )(
                            minimum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[0]
                            ),
                            maximum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[1]
                            ),
                            boundary=tmnre_parser[f"{prior_type} DISTRIBUTIONS"][
                                key
                            ].split(",")[1],
                        )
                    else:
                        ext_priors[key] = getattr(
                            prior_core,
                            tmnre_parser[f"{prior_type} DISTRIBUTIONS"][key].split(",")[
                                0
                            ],
                        )(
                            minimum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[0]
                            ),
                            maximum=float(
                                tmnre_parser[f"{prior_type} PRIORS"][key].split(",")[1]
                            ),
                            boundary=None,
                        )
    for key in int_priors.keys():
        int_priors[key].minimum = float(tmnre_parser["INT PRIORS"][key].split(",")[0])
        int_priors[key].maximum = float(tmnre_parser["INT PRIORS"][key].split(",")[1])

    for key in ext_priors.keys():
        if (
            key in ["geocent_time"]
            and "geocent_time" in tmnre_parser["INT PRIORS"].keys()
        ):
            ext_priors[key].minimum = float(
                tmnre_parser["INT PRIORS"][key].split(",")[0]
            )
            ext_priors[key].maximum = float(
                tmnre_parser["INT PRIORS"][key].split(",")[1]
            )
        elif key not in ["geocent_time"]:
            ext_priors[key].minimum = float(
                tmnre_parser["EXT PRIORS"][key].split(",")[0]
            )
            ext_priors[key].maximum = float(
                tmnre_parser["EXT PRIORS"][key].split(",")[1]
            )

    return int_priors, ext_priors
