# Setting up a *PEREGRINE* configuration file

### **Source properties:** 
```
[SOURCE]
source_type = BBH
aligned_spins = False
```
- `source_type` | Type: `str` | Choice of CBC source type 
    - `BBH` (Binary black hole)
    - `NSBH` (Neutron star black hole)
    - `BNS` (Binary neutron star)
- `aligned_spins` | Type: `bool` | Choice of aligned or precessing CBC

### **Waveform generator parameters:**
```
[WAVEFORM PARAMS]
sampling_frequency = 2048
duration = 4
start_offset = 2
start = -2
waveform_apprx = SEOBNRv4PHM
minimum_frequency = 20
maximum_frequency = 1024
reference_frequency = 50
ifo_list = H1,L1,V1
ifo_noise = True
```
- `sampling_frequency` | Type: `int` [Hz] | Sampling frequency of the GW signal
- `duration` | Type: `int` [s] | Duration of the GW signal
- `start_offset` | Type: `float` [s] | Offset time in seconds from start of detector window to merger time
-  `start` | Type: `float` [s] | Start time in GPS seconds of the detector window
- `waveform_apprx` | Type: `str` | Waveform approximant (from `lalsim`) to be used for waveform generation
    - e.g. `SEOBNRv4PHM`, `IMRPhenomXPHM`, `IMRPhenomTPHM`, `IMRPhenomPV2`
- `minimum_frequency` | Type: `int` [Hz] | Low frequency limit of the GW signal to be analysed
- `maximum_frequency` | Type: `int` [Hz] | High frequency limit of the GW signal to be analysed
- `reference_frequency` | Type: `int` [Hz] | Reference frequency to be passed on to the waveform generator
- `ifo_list` | Type: `str` | interferometer names/combinations to be passed on to the bilby detector object
    - e.g. `H1,L1`
    - **Please do not leave spaces after the commas to avoid parsing errors**
- `ifo_noise` | Type: `bool` | Choice to simulate zero noise detector strains (set `True` for latest PSD noise)

### **Injection parameters:**
```
[INJECTION 1]
mass_1 = 40
mass_2 = 22
mass_ratio = 0.55
chirp_mass = 25.5983
luminosity_distance = 400.0
dec = 0.0
ra = 4.2
theta_jn = 1.7
psi = 2.5
phase = 0.5
tilt_1 = 1.5
tilt_2 = 0.2
a_1 = 0.8
a_2 = 0.2
phi_12 = 4.5
phi_jl = 1.5
geocent_time = 0.0

[INJECTION 2]
mass_1 = 50.
mass_2 = 40.
mass_ratio = 0.8
chirp_mass = 38.8838
luminosity_distance = 1200.0
dec = 0.5
ra = 1.2
theta_jn = 0.5
psi = 0.2
phase = 3.0
tilt_1 = 0.1
tilt_2 = 0.7
a_1 = 0.1
a_2 = 0.1
phi_12 = 0.5
phi_jl = 2.5
delay = 0.2
```

- These define the injection parameters for the first and second sources (with ordmerger timesering defined by the `geocent_time` and `geocent_time + delay`, `delay` is strictly greater than or equal to zero by construction)
- If you want to generate the observation, these parameters will be your ground truth.
- If you want to analyse a pre-existing GW observation, set `generate_obs` in `[TMNRE]` to `False` (In this case, the injection values are irrelevant).

### **Parameter prior limits**
```
[PRIORS 1]
mass_ratio = 0.125,1.0
chirp_mass = 25,100
theta_jn = 0.0,3.14159
phase = 0.0,6.28318
tilt_1 = 0.0,3.14159
tilt_2 = 0.0,3.14159
a_1 = 0.05,1.0
a_2 = 0.05,1.0
phi_12 = 0.0,6.28318
phi_jl = 0.0,6.28318
luminosity_distance = 100,2000
dec = -1.57079,1.57079
ra = 0.0,6.28318
psi = 0.0,3.14159
geocent_time = -0.1,0.1

[PRIORS 2]
mass_ratio = 0.125,1.0
chirp_mass = 25,100
theta_jn = 0.0,3.14159
phase = 0.0,6.28318
tilt_1 = 0.0,3.14159
tilt_2 = 0.0,3.14159
a_1 = 0.05,1.0
a_2 = 0.05,1.0
phi_12 = 0.0,6.28318
phi_jl = 0.0,6.28318
luminosity_distance = 100,2000
dec = -1.57079,1.57079
ra = 0.0,6.28318
psi = 0.0,3.14159
delay = 0.,2.
```
- Prior ranges for all the parameters of both sources
- The parameters are just given as a reference. They need not be in this order. However, the order of these parameters reflect the ordering in your final results.
- Follows `lower_bound,upper_bound` format
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Parameter prior distributions**
```
[PRIORS 1 DISTRIBUTIONS]
geocent_time = Uniform,

[PRIORS 2 DISTRIBUTIONS]
delay = Uniform,
```
- Follows `distribution_type,boundary_type` format
- Pick from standard distributions used in `bilby.gw.priors` 
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Parameters defining the (`zarr`) store for the waveform simulations**
```
[ZARR PARAMS]
run_id = overlapping_example
use_zarr = True
sim_schedule = 30_000,60_000,60_000,120_000,120_000,150_000,150_000,150_000
chunk_size = 1000
run_parallel = True
njobs = 16
targets = z_int,z_ext,z_total,d_t,d_f,d_f_w,n_t,n_f,n_f_w
store_path = tmnre_store/overlapping_example
run_description = Full test of overlapping sources
```
- `run_id` | Type: `str` | Unique identifier for the **peregrine** run (names the output directory and result files)
- `use_zarr` | Type: `bool` | Option to use a zarr store for storing simulations (recommended setting: `True`)
- `sim_schedule` | Type: `int` | Schedule for number of simulations per round of **peregrine**-tmnre 
    - Follows : `n_sims_R1`,`n_sims_R2`,...,`n_sims_RN` where `N` is the number of rounds
- `chunk_size` | Type: `int` | Number of simulations to generate per batch
- `run_parallel` | Type: `bool` | Option to simulate parallely across cpus
- `njobs` | Type: `int` | number of parallel simulation threads (Defaults to `n_cpus` if `njobs` > `n_cpus` of your machine)
- `targets` | Type: `str` | Targets to be simulated by the **peregrine** gw simulator
    - `z_1(2)`: Parameter samples from the prior for the first (second) source
    - `z_total`: This is `z_1` and `z_2` concatenated (Not sampled independently)
    - `d_t`: Time domain detector strain with zero noise
    - `d_f`: Frequency domain detector strain with zero noise
    - `d_f_w`: Whitened frequency domain detector strain with zero noise
    - `n_t`: Time domain detector noise strain (generated with aLIGO O4 PSD)
    - `n_f`: Frequecy domain detector noise strain (generated with aLIGO O4 PSD)
    - `n_f_w`: Whitened frequency domain detector noise strain (generated with aLIGO O4 PSD)
- `store_path` | Type: `str` | Path to the directory to store the simulations
- `run_description` | Type: `str` | Description for the specific **peregrine** run
- **Please do not leave spaces after the commas to avoid parsing errors**

### **TMNRE parameters**
```
[TMNRE]
num_rounds = 8
1d_only = True
infer_only = True
marginals = (0,1)
bounds_th = 1e-5
resampler = False
shuffling = True
noise_targets = n_t,n_f_w
generate_obs = False
obs_path = tmnre_store/test_new_install/observation_test_new_install
```
- `num_rounds` | Type: `int` | Number of TMNRE rounds to be executed
- `1d_only` | Type: `bool` | Choice of training only the 1D marginals (if `True`, neglects the `marginals` argument)
- `infer_only` | Type: `bool` | Choice for running only inference if you have a pretrained NN
- `marginals` | Type: `tuple` | If `1d_only` is set to `False`, specify the 2D marginals that you want to train
- `bounds_th` | Type: `float` | Threshold determining the bounds for each round of truncation. ($\epsilon$ defined in [Cole et al.](https://arxiv.org/abs/2111.08030))
- `resample` | Type: `bool` | Choice for resampling the noise realizations at each training iteration (slow!)
- `shuffling` | Type: `bool` | Choice of shuffling the noise realizations within taining batches (fast!). Same purpose as the noise resampler but faster if you have the noise strains sampled along with the simulations (See `targets` in `[ZARR PARAMS]`) 
- `noise_targets` | Type: `str` | Noise targets to be used (Should comply with the data strains used for training (Default: `n_t`,`n_f_w`)
- `generate_obs` | Type: `bool` | Choice to generate the observation before training
- `obs_path` | Type: `str` | Path to observation file (loaded as a pickle object) when `generate_obs` is `False`

### **Hyperparameters for training the NN**
```
[HYPERPARAMS]
min_epochs = 1
max_epochs = 100
early_stopping = 7
learning_rate = 5e-4
num_workers = 8
training_batch_size = 128
validation_batch_size = 128
train_data = 0.9
val_data = 0.1
```
- `min_epochs` | Type: `int` | Minimum number of epochs to train for
- `max_epochs` | Type: `int` | Maximum number of epochs to train for
- `early_stopping` | Type: `int` | Number of training epochs to wait before stopping training in case of overfitting (reverts to the last minimum validation loss epoch)
- `learning_rate` | Type: `float` | The initial learning rate of the trainer
- `num_workers` | Type: `int` | Number of worker processes for loading training and validation data
- `training_bath_size` | Type: `int` | Batch size of the training data to be passed on to the dataloader
- `validation_bath_size` | Type: `int` | Batch size of the validation data to be passed on to the dataloader
- `train_data` | Type: `float` | $\in$ [0,1], fraction of simulation data to be used for training
- `min_epochs` | Type: `float` | $\in$ [0,1], fraction of simulation data to be used for validation/testing

### **Device parameters for training the NN**
```
[DEVICE PARAMS]
device = gpu
n_devices = 1
```
- `device` | Type: `str` | Device on which training is executed (Choice between `gpu` or `cpu`)
- `n_devices` | Type: `int` | Number of devices that the training can be parallelized over
---
