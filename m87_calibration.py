import ehtim as eh
import numpy as np

def calibrate(obs):

    #-------------------------------------------------------------------------------
    # Fiducial imaging parameters obtained from the eht-imaging parameter survey
    #-------------------------------------------------------------------------------
    zbl        = 0.60               # Total compact flux density (Jy)
    prior_fwhm = 40.0*eh.RADPERUAS  # Gaussian prior FWHM (radians)
    sys_noise  = 0.02               # fractional systematic noise
                                    # added to complex visibilities

    # constant regularization weights
    reg_term  = {'simple' : 100,    # Maximum-Entropy
                'tv'     : 1.0,    # Total Variation
                'tv2'    : 1.0,    # Total Squared Variation
                'l1'     : 0.0,    # L1 sparsity prior
                'flux'   : 1e4}    # compact flux constraint

    # initial data weights - these are updated throughout the imaging pipeline
    data_term = {'amp'    : 0.2,    # visibility amplitudes
                'cphase' : 1.0,    # closure phases
                'logcamp': 1.0}    # log closure amplitudes


    #-------------------------------------------------------------------------------
    # Fixed imaging parameters
    #-------------------------------------------------------------------------------
    # obsfile   = args.infile         # Pre-processed observation file
    ttype     = 'direct'              # Type of Fourier transform ('direct', 'nfft', or 'fast')
    npix      = 64                  # Number of pixels across the reconstructed image
    fov       = 128*eh.RADPERUAS    # Field of view of the reconstructed image
    maxit     = 100                 # Maximum number of convergence iterations for imaging
    stop      = 1e-4                # Imager stopping criterion
    gain_tol  = [0.02,0.2]          # Asymmetric gain tolerance for self-cal; we expect larger values
                                    # for unaccounted sensitivity loss
                                    # than for unaccounted sensitivity improvement
    uv_zblcut = 0.1e9               # uv-distance that separates the inter-site "zero"-baselines
                                    # from intra-site baselines
    reverse_taper_uas = 5.0         # Finest resolution of reconstructed features

    # Specify the SEFD error budget
    # (reported in First M87 Event Horizon Telescope Results III: Data Processing and Calibration)
    SEFD_error_budget = {'AA':0.10,
                        'AP':0.11,
                        'AZ':0.07,
                        'LM':0.22,
                        'PV':0.10,
                        'SM':0.15,
                        'JC':0.14,
                        'SP':0.07}

    # Add systematic noise tolerance for amplitude a-priori calibration errors
    # Start with the SEFD noise (but need sqrt)
    # then rescale to ensure that final results respect the stated error budget
    systematic_noise = SEFD_error_budget.copy()
    for key in systematic_noise.keys():
        systematic_noise[key] = ((1.0+systematic_noise[key])**0.5 - 1.0) * 0.25

    # Extra noise added for the LMT, which has much more variability than the a-priori error budget
    systematic_noise['LM'] += 0.15

    def rescale_zerobaseline(obs, totflux, orig_totflux, uv_max):
        multiplier = zbl / zbl_tot
        for j in range(len(obs.data)):
            if (obs.data['u'][j]**2 + obs.data['v'][j]**2)**0.5 >= uv_max: continue
            for field in ['vis','qvis','uvis','vvis','sigma','qsigma','usigma','vsigma']:
                obs.data[field][j] *= multiplier

    # repeat imaging with blurring to assure good convergence
    # def converge(major=3, blur_frac=1.0):
    #     for repeat in range(major):
    #         init = imgr.out_last().blur_circ(blur_frac*res)
    #         imgr.init_next = init
    #         imgr.make_image_I(show_updates=False)

    zbl_tot   = np.median(obs.unpack_bl('AA','AP','amp')['amp'])
    if zbl > zbl_tot:
        print('Warning: Specified total compact flux density ' +
            'exceeds total flux density measured on AA-AP!')

    # Flag out sites in the obs.tarr table with no measurements
    allsites = set(obs.unpack(['t1'])['t1'])|set(obs.unpack(['t2'])['t2'])
    obs.tarr = obs.tarr[[o in allsites for o in obs.tarr['site']]]
    obs = eh.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obs.data, obs.tarr,
                            source=obs.source, mjd=obs.mjd,
                            ampcal=obs.ampcal, phasecal=obs.phasecal)

    obs_orig = obs.copy() # save obs before any further modifications

    # Rescale short baselines to excize contributions from extended flux
    if zbl != zbl_tot:
        rescale_zerobaseline(obs, zbl, zbl_tot, uv_zblcut)

    # Order the stations by SNR.
    # This will create a minimal set of closure quantities
    # with the highest snr and smallest covariance.
    obs.reorder_tarr_snr()

    #-------------------------------------------------------------------------------
    # Pre-calibrate the data
    #-------------------------------------------------------------------------------

    obs_sc = obs.copy() # From here on out, don't change obs. Use obs_sc to track gain changes
    res    = obs.res()  # The nominal array resolution: 1/(longest baseline)

    # Make a Gaussian prior image for maximum entropy regularization
    # This Gaussian is also the initial image
    gaussprior = eh.image.make_square(obs_sc, npix, fov)
    gaussprior = gaussprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

    # To avoid gradient singularities in the first step, add an additional small Gaussians
    gaussprior = gaussprior.add_gauss(zbl*1e-3, (prior_fwhm, prior_fwhm, 0, prior_fwhm, prior_fwhm))

    # Reverse taper the observation: this enforces a maximum resolution on reconstructed features
    if reverse_taper_uas > 0:
        obs_sc = obs_sc.reverse_taper(reverse_taper_uas*eh.RADPERUAS)

    # Add non-closing systematic noise to the observation
    obs_sc = obs_sc.add_fractional_noise(sys_noise)

    # Make a copy of the initial data (before any self-calibration but after the taper)
    obs_sc_init = obs_sc.copy()

    # Self-calibrate the LMT to a Gaussian model
    # (Refer to Section 4's "Pre-Imaging Considerations")
    print("Self-calibrating the LMT to a Gaussian model for LMT-SMT...")

    obs_LMT = obs_sc_init.flag_uvdist(uv_max=2e9) # only consider the short baselines (LMT-SMT)
    if reverse_taper_uas > 0:
        # start with original data that had no reverse taper applied.
        # Re-taper, if necessary
        obs_LMT = obs_LMT.taper(reverse_taper_uas*eh.RADPERUAS)

    # Make a Gaussian image that would result in the LMT-SMT baseline visibility amplitude
    # as estimated in Section 4's "Pre-Imaging Considerations".
    # This is achieved with a Gaussian of size 60 microarcseconds and total flux of 0.6 Jy
    gausspriorLMT = eh.image.make_square(obs, npix, fov)
    gausspriorLMT = gausspriorLMT.add_gauss(0.6, (60.0*eh.RADPERUAS, 60.0*eh.RADPERUAS, 0, 0, 0))

    # Self-calibrate the LMT visibilities to the gausspriorLMT image
    # to enforce the estimated LMT-SMT visibility amplitude
    caltab = eh.selfcal(obs_LMT, gausspriorLMT, sites=['LM'], gain_tol=1.0,
                        method='both', ttype=ttype, caltable=True)

    # Spply the calibration solution to the full (and potentially tapered) dataset
    obs_sc = caltab.applycal(obs_sc, interp='nearest', extrapolate=True)

    return(obs_sc)