# bayesian_ring_inference

Selected collection of files for fitting and sampling geometric m-ring models on a GPU using JAX and NumPyro. This repository is a work in progress.

- ```numpyro_sampler.py``` contains the main script for sampling and fitting.
- Slides for a presentation related to this work are accessible at https://docs.google.com/presentation/d/1SdJtdU2SyUm-1JhMOfeH_VTbPXfRNwtUt8_cz42T0XA/edit?usp=sharing
- ```jaxperiments.ipynb``` contains a small selection of sample JAX computations, roughly in accordance with the material in the above presentation. This file is also accessible via Google Colab: https://colab.research.google.com/github/saopeter/bayesian_ring_inference/blob/main/jaxperiments.ipynb

## Usage: 

Running numpyro_sampler.py with all default values will use NUTS to fit visibility amplitudes and closure phases for an m-ring with 2 beta parameters to simulated data obtained via eht-imaging. 

More generally, this script can be used to fit an arbitrary m-ring model to data. Parameters intended to be user-modifiable are defined at the top of the file. Commonly modified parameters are:

- priors, a dictionary defining uniform prior bounds on F0 (total flux) in Janskys, and w (full width half max), d (ring diameter), x0 and y0 (location of the center of the ring) the latter four having units of &mu;arcseconds. Additionally, the number of beta parameters can be specified as a prior. Each beta parameter is sampled using the range \[0, 1] for its amplitude and \[-pi, pi) for its phase.
- fit, a dictionary that defines how we're fitting our model. Options are visibilities, visibility amplitudes, closure phases, and log closure amplitudes. These can be chosen in arbitrary combinations.
- fit_target controls whether we generate and fit simulated data, or fit April 11 2017 M87* observations.
- params, a dictionary containing the underlying truth values for model parameters when creating and fitting simulated data.
- num_warmup and num_samples, which control the length of our NUTS runs.

By default, JAX and NumPyro will execute sampling on a TPU or GPU unless explicitly told to use a specific platform. For this, open a Terminal and set the environment variables: 

```export JAX_PLATFORMS="cpu"```

or

```export JAX_PLATFORMS="gpu"```

according to your needs.

Once NUTS sampling has concluded, (unweighted) samples can be obtained via mcmc.get_samples() object. You can generate trace and posterior plots using ```plot_posterior(mcmc.get_samples())``` and ```plot_trace(mcmc.get_samples())```. You can additionally plot the mean sample in the image domain using ```params_to_image(get_means(mcmc.get_samples()))```. The convenience function ```plot_all``` combines all of these. Chi squared values for parameter fits can be computed using the provided helper validation functions.

## Known issues:

- Fitting log closure amplitudes and closure phases to models with 2 or more beta parameters can sometimes return unexpected values.
- For sampling beta parameters we use a Von Mises distribution with a circular reparametrization that maps inputs from the real line to the range \[-pi, pi) to avoid issues sampling at the boundary. This causes NumPyro to internally create new variables of the form "<variable>_unwrapped" and report those in its run summaries. Sometimes the values reported by the unwrapped sampling variables are well outside the expected [-pi, pi) range.
- Bessel functions of the first kind are unavailable in jax.scipy.special as of the time of this writing. To work around this, we use the integral form of the Bessel functions and numerically integrate them when necessary. This carries a performance penalty compared with the usual implementation. 
