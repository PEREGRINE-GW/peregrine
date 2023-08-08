import numpy as np
from datetime import datetime
import swyft
import swyft.lightning as sl
import bilby
from bilby.gw import WaveformGenerator, detector

bilby.core.utils.logger.setLevel("WARNING")


class Simulator(sl.Simulator):
    def __init__(self, conf, bounds=None):
        super().__init__()
        self.injection_parameters_1 = conf["injection_1"].copy()
        self.injection_parameters_2 = conf["injection_2"].copy()
        self.waveform_arguments = conf["waveform_params"]
        self.ifo_list = conf["waveform_params"]["ifo_list"]
        self.fd_source_model = conf["source"]["fd_source_model"]
        self.param_conversion_model = conf["source"]["param_conversion_model"]
        self.priors = conf["priors"]
        self.priors_1 = self.priors["priors_1"]
        self.priors_2 = self.priors["priors_2"]
        if bounds is not None:
            self.bounds = bounds
            for key in self.priors_1.keys():
                self.priors_1[key].minimum = self.bounds[
                    conf["param_idxs"][key + "_1"]
                ][0]
                self.priors_1[key].maximum = self.bounds[
                    conf["param_idxs"][key + "_1"]
                ][1]
            for key in self.priors_2.keys():
                self.priors_2[key].minimum = self.bounds[
                    conf["param_idxs"][key + "_2"]
                ][0]
                self.priors_2[key].maximum = self.bounds[
                    conf["param_idxs"][key + "_2"]
                ][1]
        waveform_generator = WaveformGenerator(
            duration=self.waveform_arguments["duration"],
            start_time=self.waveform_arguments["start"],
            sampling_frequency=self.waveform_arguments["sampling_frequency"],
            frequency_domain_source_model=self.fd_source_model,
            parameter_conversion=self.param_conversion_model,
            waveform_arguments=self.waveform_arguments,
        )
        self.waveform_generator = waveform_generator
        self.transform_samples = swyft.to_numpy32

    def generate_observation(self):
        params_1 = self.injection_parameters_1.copy()
        params_2 = self.injection_parameters_2.copy()
        z_1 = np.array([params_1[key] for key in self.priors_1.keys()])
        z_2 = np.array([params_2[key] for key in self.priors_2.keys()])
        return self.sample(
            conditions={
                "z_1": z_1,
                "z_2": z_2,
            },
            targets=["d_t", "d_f_w", "d_f", "n_t", "n_f_w", "n_f"],
        )

    def sample_priors_1(self):
        z_1 = np.array([self.priors_1[key].sample() for key in self.priors_1.keys()])
        return z_1

    def sample_priors_2(self):
        z_2 = np.array([self.priors_2[key].sample() for key in self.priors_2.keys()])
        return z_2

    def generate_z_total(
        self,
        z_1,
        z_2,
    ):
        return np.concatenate(
            (
                z_1,
                z_2,
            )
        )

    def build_param_dict(
        self,
        z_1,
        z_2,
    ):
        params_1 = self.injection_parameters_1.copy()
        params_2 = self.injection_parameters_2.copy()
        for idx, key in enumerate(self.priors_1.keys()):
            params_1[key] = z_1[idx]
        for idx, key in enumerate(self.priors_2.keys()):
            params_2[key] = z_2[idx]
        params_2["geocent_time"] = params_1["geocent_time"] + params_2["delay"]
        params_2.pop("delay")
        return (
            params_1,
            params_2,
        )

    def generate_d_t(
        self,
        z_1,
        z_2,
    ):
        params_1, params_2 = self.build_param_dict(
            z_1,
            z_2,
        )
        det = detector.InterferometerList(self.ifo_list)
        det.set_strain_data_from_zero_noise(
            sampling_frequency=self.waveform_arguments["sampling_frequency"],
            duration=self.waveform_arguments["duration"],
            start_time=self.waveform_generator.start_time,
        )
        det.inject_signal(
            waveform_generator=self.waveform_generator, parameters=params_1
        )
        det.inject_signal(
            waveform_generator=self.waveform_generator, parameters=params_2
        )
        return 1e21 * np.vstack([ifo.time_domain_strain for ifo in det])

    def generate_d_f(
        self,
        z_1,
        z_2,
        # z_3
    ):
        params_1, params_2 = self.build_param_dict(
            z_1,
            z_2,
        )

        # , params_3
        det = detector.InterferometerList(self.ifo_list)
        det.set_strain_data_from_zero_noise(
            sampling_frequency=self.waveform_arguments["sampling_frequency"],
            duration=self.waveform_arguments["duration"],
            start_time=self.waveform_generator.start_time,
        )
        det.inject_signal(
            waveform_generator=self.waveform_generator, parameters=params_1
        )
        det.inject_signal(
            waveform_generator=self.waveform_generator, parameters=params_2
        )
        d_f = np.vstack(
            [
                np.vstack(
                    (ifo.frequency_domain_strain.real, ifo.frequency_domain_strain.imag)
                )
                for ifo in det
            ]
        )
        return d_f

    def generate_d_f_w(
        self,
        z_1,
        z_2,
    ):
        params_1, params_2 = self.build_param_dict(
            z_1,
            z_2,
        )
        det = detector.InterferometerList(self.ifo_list)
        det.set_strain_data_from_zero_noise(
            sampling_frequency=self.waveform_arguments["sampling_frequency"],
            duration=self.waveform_arguments["duration"],
            start_time=self.waveform_generator.start_time,
        )
        det.inject_signal(
            waveform_generator=self.waveform_generator, parameters=params_1
        )
        det.inject_signal(
            waveform_generator=self.waveform_generator, parameters=params_2
        )
        d_f_w = np.vstack(
            [
                np.vstack(
                    (
                        ifo.whitened_frequency_domain_strain.real,
                        ifo.whitened_frequency_domain_strain.imag,
                    )
                )
                for ifo in det
            ]
        )
        return d_f_w

    def generate_noise(self):
        det = detector.InterferometerList(self.ifo_list)
        det.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.waveform_arguments["sampling_frequency"],
            duration=self.waveform_arguments["duration"],
            start_time=self.waveform_generator.start_time,
        )
        n_t = np.vstack([ifo.time_domain_strain for ifo in det])
        n_f = np.vstack(
            [
                np.vstack(
                    (ifo.frequency_domain_strain.real, ifo.frequency_domain_strain.imag)
                )
                for ifo in det
            ]
        )
        n_f_w = np.vstack(
            [
                np.vstack(
                    (
                        ifo.whitened_frequency_domain_strain.real,
                        ifo.whitened_frequency_domain_strain.imag,
                    )
                )
                for ifo in det
            ]
        )
        return np.array([n_t, n_f, n_f_w], dtype=object)

    def generate_n_t(self, noise):
        return 1e21 * noise[0]

    def generate_n_f(self, noise):
        return noise[1]

    def generate_n_f_w(self, noise):
        return noise[2]

    def build(self, graph):
        """
        Define the computational graph, which allows us to sample specific targets efficiently
        """
        z_1 = graph.node("z_1", self.sample_priors_1)
        z_2 = graph.node("z_2", self.sample_priors_2)
        z_total = graph.node(
            "z_total",
            self.generate_z_total,
            z_1,
            z_2,
        )
        d_t = graph.node(
            "d_t",
            self.generate_d_t,
            z_1,
            z_2,
        )
        d_f = graph.node(
            "d_f",
            self.generate_d_f,
            z_1,
            z_2,
        )
        d_f_w = graph.node(
            "d_f_w",
            self.generate_d_f_w,
            z_1,
            z_2,
        )
        noise = graph.node("n", self.generate_noise)
        n_t = graph.node("n_t", self.generate_n_t, noise)
        n_f = graph.node("n_f", self.generate_n_f, noise)
        n_f_w = graph.node("n_f_w", self.generate_n_f_w, noise)


def init_simulator(conf: dict, bounds=None):
    """
    Initialise the swyft simulator
    Args:
      conf: dictionary of config options, output of init_config
      bounds: bounds object, output of load_bounds
    Returns:
      Swyft lightning simulator instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    """
    simulator = Simulator(conf, bounds)
    return simulator


def simulate(simulator, store, conf, max_sims=None):
    """
    Run a swyft simulator to save simulations into a given zarr store
    Args:
      simulator: swyft simulator object
      store: initialised zarr store
      conf: dictionary of config options, output of init_config
      max_sims: maximum number of simulations to perform (otherwise will fill store)
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> simulate(simulator, store, conf)
    """
    store.simulate(
        sampler=simulator,
        batch_size=int(conf["zarr_params"]["chunk_size"]),
        max_sims=max_sims,
    )
