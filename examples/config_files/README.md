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
[INJECTION]
mass_1 = 39.53573995392598
mass_2 = 34.8722208102226
mass_ratio = 0.8857620985418904
chirp_mass = 32.136969061169324
luminosity_distance = 900
dec = 0.07084716171380845
ra = 5.555599820502261
theta_jn = 0.44320777946320117
psi = 1.0995170458005799
phase = 5.089358282766109
tilt_1 = 1.4974326044527126
tilt_2 = 1.1019600169566186
a_1 = 0.9701993491043245
a_2 = 0.8117959745751914
phi_12 = 6.220246980963511
phi_jl = 1.884805935473119
geocent_time = 0.0
```

- If you want to generate the observation, these parameters will be your ground truth.
- If you want to analyse a pre-existing GW observation, set `generate_obs` in `[TMNRE]` to `False` (In this case, the injection values are irrelevant).

### **Intrinsic parameter prior limits**
```
[INT PRIORS]
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
```
- The intrinsic vs extrinsic parameters are just given as a reference. They need not be in this order. However, the order of these parameters reflect the ordering in your final results.
- Follows `lower_bound,upper_bound` format
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Intrinsic parameter prior distributions**
```
[INT DISTRIBUTIONS]
mass_ratio = Uniform,reflective
chirp_mass = Uniform,reflective
```
- Follows `distribution_type,boundary_type` format
- Pick from standard distributions used in `bilby.gw.priors` 
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Extrinsic parameter prior limits**
```
[EXT PRIORS]
luminosity_distance = 100,1500
dec = -1.57079,1.57079
ra = 0.0,6.28318
psi = 0.0,3.14159
geocent_time = -0.1,0.1
```
- Follows `lower_bound,upper_bound` format
- **Please do not leave spaces after the commas to avoid parsing errors**
### **Extrinsic parameter prior distributions**
```
[EXT DISTRIBUTIONS]
geocent_time = Uniform,
```
### **Parameters defining the (`zarr`) store for the waveform simulations**
```
[ZARR PARAMS]
run_id = test_new_install
use_zarr = True
sim_schedule = 30_000,60_000,60_000,120_000,120_000,150_000,150_000,150_000
chunk_size = 1000
run_parallel = True
njobs = 16
targets = z_int,z_ext,z_total,d_t,d_f,d_f_w,n_t,n_f,n_f_w
store_path = tmnre_store/test_new_install
run_description = Example run for a 15D precessing CBC with 3 LVKC detectors
```
- `run_id` | Type: `str` | Unique identifier for the **peregrine** run (names the output directory and result files)
- `use_zarr` | Type: `bool` | Option to use a zarr store for storing simulations (recommended setting: `True`)
- `sim_schedule` | Type: `int` | Schedule for number of simulations per round of **peregrine**-tmnre 
    - Follows : `n_sims_R1`,`n_sims_R2`,...,`n_sims_RN` where `N` is the number of rounds
- `chunk_size` | Type: `int` | Number of simulations to generate per batch
- `run_parallel` | Type: `bool` | Option to simulate parallely across cpus
- `njobs` | Type: `int` | number of parallel simulation threads (Defaults to `n_cpus` if `njobs` > `n_cpus` of your machine)
- `targets` | Type: `str` | Targets to be simulated by the **peregrine** gw simulator
    - `z_int(_ext)`: Intrinsic(Extrinsic) parameter samples from the prior
    - `z_total`: This is `z_int` and `z_ext` concatenated (Not sampled independently)
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


### **Standard likelihood based sampler parameters (for comparison tests)**
```
[SAMPLING]
sampler = dynesty
npoints = 2000
nsamples = 2000
printdt = 5
walks = 100
nact = 10
ntemps = 10
nlive = 2000
nwalkers = 10
distance = False
time = False
phase = True
resume_from_ckpt = True
```
- `sampler` | Type: `str` | Choice of sampler from available samplers ran through `bilby`
    - e.g. : `dynesty`, `cpnest`, `pymultinest`, `ptemcee`, `bilby_mcmc`,
- Requires the sampler to be installed on your system
- `resume_from_ckpt` | Type: `bool` | Choice to resume interrupted bilby inference run from a checkpoint dump file

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
