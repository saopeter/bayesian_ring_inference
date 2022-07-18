import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from jax import random

from m87_calibration import calibrate

eht = eh.array.load_txt('EHT2019.txt')
fit_target = "simulated" #"M87" or "simulated"

# # Circular Gaussian model
# #units of microarcseconds
# dx = -15
# dy = 20
# h = 40
# #units of janskys
# f = 1.3
# params = {"dx":dx, "dy":dy, "h":h, "f":f}

# print(f'underlying model parameters: dx:{dx:.2e}, dy:{dy:.2e}, h:{h:.2e}, f:{f:.2e}\n')

# #eht-imaging likes units of radians, so we convert all our non-flux parameters
# model_type = "circ_gauss"
# model = model.add_circ_gauss(f, h*eh.RADPERUAS, x0=dx*eh.RADPERUAS, y0=dy*eh.RADPERUAS)

# Basic M-ring model
#units of janskys:
model_type = "mring"
F0 = 1.3 
#units of microarcseconds:
d = 50
w = 20
x0 = 0
y0 = 0
#dimensionless (complex):
betas_amplitudes = [.5]
betas_phases = [jnp.pi]

params = {"F0":F0, "d":d, "w":w, "x0":x0, "y0":y0, "betas_amplitudes":betas_amplitudes, "betas_phases":betas_phases}


# Fitting parameters
priors = {"F0":(0, 10), "d": (5, 60), "w": (10, 30), "betas": 1} # F0, d, w uniform; betas is a number of parameters, each of which is uniform in (0,1) and (-pi to pi) in phase
fit = {"visibilities": False, "visibility amplitudes": True, "closure phases": True, "log closure amplitudes": False}

# Helper function for converting between ehtim mring and my mring model parameters
def convert_params_to_ehtim(params, model_type=model_type):
    ehtim_params = {}
    if model_type == 'mring':
        # my params: F0, d (UAS), w (UAS), beta_amplitudes(real list), beta_phases(rad list)
        # eht params: F0, d (rad), alpha (rad), x0 (rad), y0 (rad), beta_list (complex list)
        betas_amplitudes = params['betas_amplitudes']
        betas_phases = params['betas_phases']

        ehtim_params['F0'] = params['F0']
        ehtim_params['d'] = params['d']*eh.RADPERUAS
        ehtim_params['alpha'] = params['w']*eh.RADPERUAS
        ehtim_params['x0'] = 0
        ehtim_params['y0'] = 0
        betas = [ betas_amplitudes[i]*jnp.exp((1j*betas_phases[i])) for i in range(len(betas_amplitudes)) ]
        ehtim_params['beta_list'] = betas

    return ehtim_params

# Data we're fitting against -- either 2017 M87 Stokes I uvfits data, or simulated
obs = None
model = None
# M87 data
if fit_target == "M87":
    obs = eh.obsdata.load_uvfits('SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits')
    obs.add_scans()
    obs_sa = obs.avg_coherent(0.,scan_avg=True)

    obs_sc = calibrate(obs_sa)
    obs = obs_sc

# #Simulated data
elif fit_target == "simulated":

    # the following parameters were added during the 4/28 meeting:
    # changes from the usual parameters essentially inject more noise into the model
    # tint_sec = .5
    # tadv_sec = 3600
    # tstart_hr = 0
    # tstop_hr = 24
    # bw_hz = 1e7

    #parameters from 7/7 meeting:
    # tint_sec = 10
    # tadv_sec = 3600
    # tstart_hr = 0
    # tstop_hr = 24
    # bw_hz = 4e8

    # parameters from example modeling script
    tint_sec = 5
    tadv_sec = 3600
    tstart_hr = 0
    tstop_hr = 24
    bw_hz = 1e9

    # Model parameters
    model = eh.model.Model()
    ehtim_params = convert_params_to_ehtim(params, model_type=model_type)
    model = model.add_thick_mring(**ehtim_params)

    # obs = model.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, ampcal=True, phasecal=True,seed=4)
    obs = model.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, ampcal=True, phasecal=False,seed=7)
    obs = obs.add_fractional_noise(.1)
    obs = model.observe_same_nonoise(obs)

def generate_circ_gauss_model(dx, dy, h, f):
    # re-convert from microarcseconds back to radians
    dx = dx*eh.RADPERUAS
    dy = dy*eh.RADPERUAS
    h = h*eh.RADPERUAS
    def forward_model(u, v):
        return f * jnp.exp(-jnp.pi**2/(4.*jnp.log(2.)) * (u**2 + v**2) * h**2)\
               * jnp.exp(1j * 2.0 * jnp.pi * (u * dx + v * dy))
    return forward_model

# replacement for scipy.special.jv, which is not available in jax
# computed via trapz using the integral definition of J
def bessel_j(n, z, num_samples=100):
    # print("z", z, z.shape, "n", n, n.shape)
    z = jnp.asarray(z)
    scalar = z.ndim == 0
    if scalar:
        z = z[np.newaxis]
    z = z[:, np.newaxis]
    tau = np.linspace(0, jnp.pi, num_samples)
    integrands = jnp.trapz(jnp.cos(n*tau - z*jnp.sin(tau)), x=tau)
    if scalar:
        return (1./jnp.pi)*integrands.squeeze()
    return (1./jnp.pi)*integrands

# As described in 2022 paper 4 section 4.3: https://iopscience.iop.org/article/10.3847/2041-8213/ac6736/pdf
# This consists of an infinitesimally thin ring convolved with a Gaussian kernel
# parameters: F_ring, d_ring, W_kernel (FWHM), plus 2m beta parameters. Assume ring is centered at (0,0) in the image domain
# betas are not always magnitude 0<x<1 complex numbers

# 1, 1 betas eg.1 (note: this produces negative images)
# 1, 1/root2 1/root2i eg. 2

# TODO: add x0 and y0 parameters
def generate_m_ring_model(F0, d, w, betas_amplitudes, betas_phases, x0=0, y0=0):
    d = d*eh.RADPERUAS
    w = w*eh.RADPERUAS
    betas = jnp.asarray( [ betas_amplitudes[i]*jnp.exp((1j*betas_phases[i])) for i in range(len(betas_amplitudes)) ])
    def forward_model(u, v):
        mag = jnp.sqrt(u**2+ v**2)
        phi = jnp.arctan2(u,v)
        # phi += jnp.pi
        extended_betas = jnp.concatenate( (jnp.flip(jnp.conj(betas)), jnp.array([1]), betas )) #[complex conjugates of betas, 1, betas]
        ks = jnp.arange(-len(betas), len(betas)+1)
        exps = jnp.array([jnp.exp(1j*n*(phi-jnp.pi/2.)) for n in ks]).T
        # exps = jnp.array([jnp.exp(1j*n*(phi)) for n in ks]).T
        Js = jnp.array([bessel_j(n, jnp.pi*mag*d) for n in ks]).T
        V_ring = F0*jnp.sum(extended_betas*Js*exps, axis=1)
        V_ring = jnp.exp((-jnp.pi**2*w**2*mag**2)/(4.*jnp.log(2.)))*V_ring
        return jnp.conj(V_ring)
    return forward_model

# def generate_m_ring_model_test(F0, d, w, betas_amplitudes, betas_phases, x0=0, y0=0):
#         phi = jnp.angle(v + 1j*u)
#         # Flip the baseline sign to match eht-imaging conventions
#         phi += np.pi
#         z = np.pi * params['d'] * (u**2 + v**2)**0.5
#         beta_factor = (sps.jv(0, z) 
#             + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
#             + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))

#         val = (params['F0'] * beta_factor
#                * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
#                * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 

def gaussian_model():
    dx = numpyro.sample("dx", dist.Uniform(low=-40, high=40))
    dy = numpyro.sample("dy", dist.Uniform(low=-40, high=40))
    h = numpyro.sample("h", dist.Uniform(low=0, high=80))
    f = numpyro.sample("f", dist.Uniform(low=0, high=10))
    ft = generate_circ_gauss_model(dx, dy, h, f)

    u = obs.data['u']
    v = obs.data['v']
    vis = obs.data['vis']
    sigma = obs.data['sigma']

    # Fit real and imaginary parts separately.
    numpyro.sample("re(obs)", dist.Normal(ft(u,v).real, sigma), obs = vis.real)
    numpyro.sample("im(obs)", dist.Normal(ft(u,v).imag, sigma), obs = vis.imag)

# @numpyro.handlers.reparam(config={"closure phases": numpyro.infer.reparam.CircularReparam()})
def mring_model(obs=obs, priors=priors, fit=fit):

    # parameters: F0, d, w, betas
    # distributions are uniform; betas are specified as a number

    F0 = numpyro.sample("F0", dist.Uniform(low=priors["F0"][0], high=priors["F0"][1]))
    d = numpyro.sample("d", dist.Uniform(low=priors["d"][0], high=priors["d"][1]))
    w = numpyro.sample("w", dist.Uniform(low=priors["w"][0], high=priors["w"][1]))
    betas_amplitudes = []
    betas_phases = []
    # TODO: use numpyro plate notation? 
    for i in range(priors["betas"]):
        beta_amp = numpyro.sample("abs(b"+str(i+1)+")", dist.Uniform(low=0, high=1))
        #TODO: encode circularity of phases for better sampling of higher order beta parameters
        beta_phase = numpyro.sample("angle(b"+str(i+1)+")", dist.Uniform(low=0, high=2*jnp.pi))
        # beta_phase = numpyro.sample("angle(b"+str(i+1)+")", dist.VonMises(0, 0))
        betas_amplitudes.append(beta_amp)
        betas_phases.append(beta_phase)

    ft = generate_m_ring_model(F0=F0, d=d, w=w, betas_amplitudes=betas_amplitudes, betas_phases=betas_phases)

    if fit["visibilities"]:
        u = obs.data['u']
        v = obs.data['v']
        vis = obs.data['vis']
        vis_sigma = obs.data['sigma']
        numpyro.sample("Re(obs)", dist.Normal(ft(u,v).real, vis_sigma), obs = vis.real)
        numpyro.sample("Im(obs)", dist.Normal(ft(u,v).imag, vis_sigma), obs = vis.imag)

    if fit["visibility amplitudes"]:
        u = obs.data['u']
        v = obs.data['v']
        vis_amp = obs.unpack(['amp'],debias=True)['amp']
        vis_sigma = obs.data['sigma']
        numpyro.sample("visibility amplitudes", dist.Normal(abs(ft(u,v)), vis_sigma), obs = vis_amp)

    if fit["closure phases"]:
        # TODO: refactor these sections, this code is going to be similar across all models
        #Pull out closure phases from simulated data
        cphase_data = obs.c_phases(ang_unit='rad')
        cphases = cphase_data['cphase']
        cphases_sigmas = cphase_data['sigmacp']

        # Calculate our model closure phases
        cphases_u1 = cphase_data['u1']
        cphases_u2 = cphase_data['u2']
        cphases_u3 = cphase_data['u3']
        cphases_v1 = cphase_data['v1']
        cphases_v2 = cphase_data['v2']
        cphases_v3 = cphase_data['v3']

        cphase12 = jnp.angle(ft(cphases_u1, cphases_v1))
        cphase23 = jnp.angle(ft(cphases_u2, cphases_v2))
        cphase31 = jnp.angle(ft(cphases_u3, cphases_v3))

        model_cphase = cphase12 + cphase23 + cphase31
        reparameterized_cphase = jnp.remainder(model_cphase + jnp.pi, 2*jnp.pi) - jnp.pi
        numpyro.sample("closure phases", dist.VonMises(reparameterized_cphase, 1/cphases_sigmas**2), obs = cphases)

    if fit["log closure amplitudes"]:
        # Pull out closure amplitudes from simulated data
        logcamp = obs.c_amplitudes(ctype='logcamp', debias=True)
        lcamps = logcamp['camp']
        lcamps_sigmas = logcamp['sigmaca']

        # Calculate our model log closure amplitudes
        lcamps_u1 = logcamp['u1']
        lcamps_u2 = logcamp['u2']
        lcamps_u3 = logcamp['u3']
        lcamps_u4 = logcamp['u4']
        lcamps_v1 = logcamp['v1']
        lcamps_v2 = logcamp['v2']
        lcamps_v3 = logcamp['v3']
        lcamps_v4 = logcamp['v4']

        lcamp12 = jnp.log(jnp.abs(ft(lcamps_u1, lcamps_v1)))
        lcamp34 = jnp.log(jnp.abs(ft(lcamps_u2, lcamps_v2)))
        lcamp23 = jnp.log(jnp.abs(ft(lcamps_u3, lcamps_v3)))
        lcamp14 = jnp.log(jnp.abs(ft(lcamps_u4, lcamps_v4)))

        # Fit closure amplitudes
        model_lcamp = lcamp12 + lcamp34 - lcamp23 - lcamp14
        # print("Fitting closure amplitudes")
        numpyro.sample("log closure amplitudes", dist.Normal(model_lcamp, lcamps_sigmas), obs = lcamps)

PPL_model = None
if model_type == "circ_gauss":
    PPL_model = gaussian_model
elif model_type == "mring":
    PPL_model = mring_model

# Print information about our model and priors
if fit_target=="simulated":
    print(f"Fitting model type: {model_type} with underlying parameters: {params}")
elif fit_target=="M87":
    print(f"Fitting M87 data")
print(f"Fitting the following quantities: {','.join([k for k in fit if fit[k]])}")
print(f"Priors: {priors}")

# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

# Run NUTS.
kernel = NUTS(PPL_model)
num_warmup = 1000
num_samples = 2000
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(
    rng_key_
)
mcmc.print_summary()

#Assorted helper functions

def plot_trace(samples, model_type=None, truths=None, save=False, filename=None):
    plt.ioff()
    variables = samples.keys()
    nvars = len(variables)
    nsamples = len(samples[list(variables)[0]])
    fig, ax = plt.subplots(ncols=nvars, figsize=(2*nvars,6))
    for i, var in enumerate(variables):
        ax[i].plot(samples[var])
        if truths:
            #FIXME: fails if more than 9 beta parameters, needs better approach
            if "abs" in var:
                beta_n = int(var[-2])-1
                ax[i].axhline(truths['betas_amplitudes'][beta_n], color="red")
            elif "angle" in var:
                beta_n = int(var[-2])-1
                ax[i].axhline(truths['betas_phases'][beta_n], color="red")
            else:
                ax[i].axhline(truths[var], color="red")
        ax[i].set_ylabel(var)
    fig.supxlabel('Samples')
    fig.suptitle(f"traces, {'model: '+model_type+', ' if model_type else ''}algorithm:numpyro NUTS, samples:{nsamples}")
    fig.tight_layout()
    if save:
        if not filename:
            filename = f"numpyro_nuts_{model_type}_{nvars}params_trace"
        plt.savefig(filename)
    else:
        plt.show()

def plot_posterior(samples, model_type=None, truths=None, num_convergence=500, save=False, filename=None):
    plt.ioff()
    variables = samples.keys()
    nvars = len(variables)
    nsamples = len(samples[list(variables)[0]])
    fig, ax = plt.subplots(ncols=nvars, figsize=(2*nvars,6))
    bin_size = int(np.log(nsamples))+5
    for i, var in enumerate(variables):
        parsed_samples = np.array(samples[var][-num_convergence:]) #hist doesn't work well with jax's devicearrays
        ax[i].hist(parsed_samples, bins=bin_size)
        if truths:
            #FIXME: fails if more than 9 beta parameters, needs better approach
            if "abs" in var:
                beta_n = int(var[-2])-1
                ax[i].axvline(truths['betas_amplitudes'][beta_n], color="red")
            elif "angle" in var:
                beta_n = int(var[-2])-1
                ax[i].axvline(truths['betas_phases'][beta_n], color="red")
            else:
                ax[i].axvline(truths[var], color="red")
        ax[i].set_xlabel(var)
    fig.supxlabel('Samples')
    fig.suptitle(f"posteriors, {'model: '+model_type+', ' if model_type else ''}algorithm:numpyro NUTS, samples:{nsamples}")
    fig.tight_layout()
    if save:
        if not filename:
            filename = f"numpyro_nuts_{model_type}_{nvars}params_posterior"
        plt.savefig(filename)
    else:
        plt.show()

# params: kwargs defining model parameters for given model type
def params_to_image(params, model_type=model_type, save=False, filename=None):
    if model_type == 'mring':
        m = eh.model.Model()
        ehtim_params = convert_params_to_ehtim(params)
        m = m.add_thick_mring(**ehtim_params)
        m.source="M87*"
        if save:
            if not filename:
                filename = "fit_image"
            m.display(export_pdf=filename, show=False)
        else:
            m.display()

def get_last_sample(samples):
    variables = samples.keys()
    output = {}
    betas_amplitudes = []
    betas_phases = []
    for var in variables:
        if "abs" in var:
            n = var[-2]
            betas_amplitudes.append((samples[var][-1], n))
        elif "angle" in var:
            n = var[-2]
            betas_phases.append((samples[var][-1], n))
        else:
            output[var] = samples[var][-1]
    betas_amplitudes = sorted(betas_amplitudes, key=lambda x:x[1])
    betas_amplitudes = [x[0] for x in betas_amplitudes]
    betas_phases = sorted(betas_phases, key=lambda x:x[1])
    betas_phases = [x[0] for x in betas_phases]
    output["betas_amplitudes"] = betas_amplitudes
    output["betas_phases"] = betas_phases
    return output

def plot_all(samples=mcmc.get_samples(), model_type=None, truths=None, save=False, filename=None):
    plot_trace(samples, model_type=model_type, truths=truths, save=save, filename=filename+"_trace")
    plot_posterior(samples, model_type=model_type, truths=truths, save=save, filename=filename+"_posterior")
    params_to_image(get_last_sample(samples), save=save, filename=filename+"_image")