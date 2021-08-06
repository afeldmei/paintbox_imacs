""" Run paintbox in observed data. """
import os
import shutil
import copy

import numpy as np
from scipy import stats
import multiprocessing as mp
from astropy.table import Table, hstack, vstack
import astropy.constants as const
import emcee
import matplotlib.pyplot as plt
from tqdm import tqdm
from ppxf import ppxf_util
import seaborn as sns
import paintbox as pb
from paintbox.utils import CvD18, disp2vel
import h5py
import corner

import context

###############################################################################
# NAME:
#   EMISSION_LINES
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 7 January 2014
#   V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
#   V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numerical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014
#   V1.2.0: Perform integration over the pixels of the ppxf_util.gaussian line spread function
#       using the new function ppxf_util.gaussian(). Thanks to Eric Emsellem for the suggestion.
#       MC, Oxford, 10 August 2016
#   V1.2.1: Allow FWHM_gal to be a function of wavelength. MC, Oxford, 16 August 2016
#   V1.2.2: Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 3 August 2017
#   V1.3.0: New `tie_balmer` keyword to assume intrinsic Balmer decrement.
#       New `limit_doublets` keyword to limit ratios of [OII] & [SII] doublets.
#       New `vacuum` keyword to return wavelengths in vacuum.
#       MC, Oxford, 31 October 2017
#   V1.3.1: Account for the varying pixel size in Angstrom, when specifying the
#       weights for the Balmer series with tie_balmer=True. Many thanks to
#       Kyle Wesfall (Santa Cruz) for reporting this bug. MC, Oxford, 10 April 2018

def emission_lines(logLam_temp, lamRange_gal, FWHM_gal, pixel=True,
                   tie_balmer=False, limit_doublets=False, vacuum=False):
    """
    Generates an array of ppxf_util.gaussian emission lines to be used as gas templates in PPXF.

    Generally, these templates represent the instrumental line spread function
    (LSF) at the set of wavelengths of each emission line. In this case, pPXF
    will return the intrinsic (i.e. astrophysical) dispersion of the gas lines.

    Alternatively, one can input FWHM_gal=0, in which case the emission lines
    are delta-functions and pPXF will return a dispersion which includes both
    the intrumental and the intrinsic disperson.

    Additional lines can be easily added by editing the code of this procedure,
    which is meant as a template to be modified by the users where needed.

    For accuracy the ppxf_util.gaussians are integrated over the pixels boundaries.
    This can be changed by setting `pixel`=False.

    The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    The [OII] and [SII] doublets can be restricted to physical range of ratios.

    The Balmet Series can be fixed to the theoretically predicted decrement.

    Input Parameters
    ----------------

    logLam_temp: array_like
        is the natural log of the wavelength of the templates in Angstrom.
        ``logLam_temp`` should be the same as that of the stellar templates.
    lamRange_gal: array_like
        is the estimated rest-frame fitted wavelength range. Typically::

            lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),

        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial rough estimate of the galaxy redshift.
    FWHM_gal: float or func
        is the instrumantal FWHM of the galaxy spectrum under study in Angstrom.
        One can pass either a scalar or the name "func" of a function
        ``func(wave)`` which returns the FWHM for a given vector of input
        wavelengths.
    pixel: bool, optional
        Set this to ``False`` to ignore pixels integration (default ``True``).
    tie_balmer: bool, optional
        Set this to ``True`` to tie the Balmer lines according to a theoretical
        decrement (case B recombination T=1e4 K, n=100 cm^-3).

        IMPORTANT: The relative fluxes of the Balmer components assumes the
        input spectrum has units proportional to ``erg/(cm**2 s A)``.
    limit_doublets: bool, optional
        Set this to True to limit the rato of the [OII] and [SII] doublets to
        the ranges allowed by atomic physics.

        An alternative to this keyword is to use the ``constr_templ`` keyword
        of pPXF to constrain the ratio of two templates weights.

        IMPORTANT: when using this keyword, the two output fluxes (flux_1 and
        flux_2) provided by pPXF for the two lines of the doublet, do *not*
        represent the actual fluxes of the two lines, but the fluxes of the two
        input *doublets* of which the fit is a linear combination.
        If the two doublets templates have line ratios rat_1 and rat_2, and
        pPXF prints fluxes flux_1 and flux_2, the actual ratio and flux of the
        fitted doublet will be::

            flux_total = flux_1 + flux_1
            ratio_fit = (rat_1*flux_1 + rat_2*flux_2)/flux_total

        EXAMPLE: For the [SII] doublet, the adopted ratios for the templates are::

            ratio_d1 = flux([SII]6716/6731) = 0.44
            ratio_d2 = flux([SII]6716/6731) = 1.43.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([SII]6731_d1) = flux_1
            flux([SII]6731_d2) = flux_2

        the total flux and true lines ratio of the [SII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([SII]6716/6731) = (0.44*flux_1 + 1.43*flux_2)/flux_total

        Similarly, for [OII], the adopted ratios for the templates are::

            ratio_d1 = flux([OII]3729/3726) = 0.28
            ratio_d2 = flux([OII]3729/3726) = 1.47.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([OII]3726_d1) = flux_1
            flux([OII]3726_d2) = flux_2

        the total flux and true lines ratio of the [OII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([OII]3729/3726) = (0.28*flux_1 + 1.47*flux_2)/flux_total

    vacuum:  bool, optional
        set to ``True`` to assume wavelengths are given in vacuum.
        By default the wavelengths are assumed to be measured in air.

    Output Parameters
    -----------------

    emission_lines: ndarray
        Array of dimensions ``[logLam_temp.size, line_wave.size]`` containing
        the gas templates, one per array column.

    line_names: ndarray
        Array of strings with the name of each line, or group of lines'

    line_wave: ndarray
        Central wavelength of the lines, one for each gas template'

    """
    if tie_balmer:

        # Balmer decrement for Case B recombination (T=1e4 K, ne=100 cm^-3)
        # Table 4.4 of Dopita & Sutherland 2003 https://www.amazon.com/dp/3540433627
        # Balmer:         Htheta   Heta     Hzeta    Heps    Hdelta   Hgamma    Hbeta   Halpha
        wave = np.array([3797.90, 3835.39, 3889.05, 3970.07, 4101.76, 4340.47, 4861.33, 6562.80])  # air wavelengths
        if vacuum:
            wave = ppxf_util.air_to_vac(wave)
        gauss = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel)
        ratios = np.array([0.0530, 0.0731, 0.105, 0.159, 0.259, 0.468, 1, 2.86])
        ratios *= wave[-2]/wave  # Account for varying pixel size in Angstrom
        emission_lines = gauss @ ratios
        line_names = ['Balmer']
        w = (wave > lamRange_gal[0]) & (wave < lamRange_gal[1])
        line_wave = np.mean(wave[w]) if np.any(w) else np.mean(wave)

    else:

        # Use fewer lines here, as the weak ones are difficult to measure
        # Balmer:    Hdelta   Hgamma    Hbeta   Halpha
        line_wave = [4101.76, 4340.47, 4861.33, 6562.80]  # air wavelengths
        if vacuum:
            line_wave = ppxf_util.air_to_vac(line_wave)
        line_names = ['Hdelta', 'Hgamma', 'Hbeta', 'Halpha']
        emission_lines = ppxf_util.gaussian(logLam_temp, line_wave, FWHM_gal, pixel)


    if limit_doublets:

        # The line ratio of this doublet lam3729/lam3726 is constrained by
        # atomic physics to lie in the range 0.28--1.47 (e.g. fig.5.8 of
        # Osterbrock & Ferland 2005 https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #       -----[OII]-----
        wave = [3726.03, 3728.82]    # air wavelengths
        if vacuum:
            wave = ppxf_util.air_to_vac(wave)
        names = ['[OII]3726_d1', '[OII]3726_d2']
        gauss = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel)
        doublets = gauss @ [[1, 1], [0.28, 1.47]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

        # The line ratio of this doublet lam6716/lam6731 is constrained by
        # atomic physics to lie in the range 0.44--1.43 (e.g. fig.5.8 of
        # Osterbrock & Ferland 2005 https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #       -----[SII]-----
        wave = [6716.47, 6730.85]    # air wavelengths
        if vacuum:
            wave = ppxf_util.air_to_vac(wave)
        names = ['[SII]6731_d1', '[SII]6731_d2']
        gauss = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel)
        doublets = gauss @ [[0.44, 1.43], [1, 1]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    else:

        # Here the doublets are free to have any ratio
        #       -----[OII]-----    -----[SII]-----
        wave = [3726.03, 3728.82, 6716.47, 6730.85]  # air wavelengths
        if vacuum:
            wave = ppxf_util.air_to_vac(wave)
        names = ['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731']
        gauss = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel)
        emission_lines = np.column_stack([emission_lines, gauss])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)


    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[OIII]-----
    wave = [4958.92, 5006.84]    # air wavelengths
    if vacuum:
        wave = ppxf_util.air_to_vac(wave)
    doublet = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OIII]5007_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OI]-----
    wave = [6300.30, 6363.67]    # air wavelengths
    if vacuum:
        wave = ppxf_util.air_to_vac(wave)
    doublet = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel) @ [1, 0.33]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[OI]6300_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[NII]-----
    wave = [6548.03, 6583.41]    # air wavelengths
    if vacuum:
        wave = ppxf_util.air_to_vac(wave)
    doublet = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NII]6583_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[1])

#added by anja to ppxf_util.emission_lines version
    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[NI]-----
    wave = [5197.90, 5200.39]    # air wavelengths
    if vacuum:
        wave = ppxf_util.air_to_vac(wave)
    doublet = ppxf_util.gaussian(logLam_temp, wave, FWHM_gal, pixel) @ [1, 0.7]
    emission_lines = np.column_stack([emission_lines, doublet])
    line_names = np.append(line_names, '[NI]5200_d')  # single template for this doublet
    line_wave = np.append(line_wave, wave[1])
#----------------------

    # Only include lines falling within the estimated fitted wavelength range.
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave

###############################################################################
def make_paintbox_model(wave, store, name="test", porder=45, nssps=1,
                        sigma=100):
#    # Directory where you store your CvD models
#    base_dir = context.cvd_dir
#    # Locationg where pre-processed models will be stored for paintbox
#     store = os.path.join(context.home_dir,
#                           f"templates/CvD18_sig{sigma}_{name}.fits")
#    # Defining wavelength for templates
#     velscale = sigma / 2
#     wmin = wave.min() - 200
#     wmax = wave.max() + 50
#    # Locationg where pre-processed models will be stored for paintbox
#     store = os.path.join(context.home_dir,
#                           f"templates/CvD18_sig{sigma}_{name}_ll{round(wmin)}_{round(wmax)}_{round(velscale)}.fits")
#    """ Returns a log-rebinned wavelength dispersion with constant velocity. 
#    twave = disp2vel([wmin, wmax], velscale)
#    ssp = CvD18(twave, sigma=sigma, store=store, libpath=context.cvd_dir)
    ssp = CvD18(sigma=sigma, store=store, libpath=context.cvd_dir)
    twave = ssp.wave
    limits = ssp.limits
    if nssps > 1:
        for i in range(nssps):
            p0 = pb.Polynomial(twave, 0, pname="w")
            p0.parnames = [f"w_{i+1}"]
            s = copy.deepcopy(ssp)
            s.parnames = ["{}_{}".format(_, i+1) for _ in s.parnames]
#            print(shape(p0),shape(s))
            if i == 0:
                pop = p0 * s
            else:
                pop += (p0 * s)
    else:
        pop = ssp
    vname = "vsyst_{}".format(name)
    stars = pb.Resample(wave, pb.LOSVDConv(pop, losvdpars=[vname, "sigma"]))
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder, zeroth=True, pname=f"p{name}")
#     sed = stars * poly
#     return sed, limits

    # Including emission lines
    target_fwhm = lambda w: sigma / const.c.to("km/s").value * w * 2.355
    gas_templates, gas_names, line_wave = emission_lines(
        np.log(twave), [wave[0], wave[-1]], target_fwhm,
        tie_balmer=True, vacuum=True)
    gas_templates /= np.max(gas_templates, axis=0) # Normalize line amplitudes
    gas_names = [_.replace("_", "") for _ in gas_names]
#     for em in gas_templates.T:
#         plt.plot(twave, em)
#     plt.show()
#     print(gas_names)

    if len(gas_names) > 0:
        emission = pb.NonParametricModel(twave, gas_templates.T,
                                         names=gas_names)
        emkin = pb.Resample(wave, pb.LOSVDConv(emission,
                            losvdpars=[vname, "sigma_gas"]))
        sed = (stars + emkin) * poly
    else:
        sed = stars * poly
    return sed, limits, gas_names


def set_priors(parnames, limits, linenames, vsyst, nssps=1):
    """ Defining prior distributions for the model. """
    priors = {}
    for parname in parnames:
        name = parname.split("_")[0]
        if name in limits: #all the CvD ssp parameters
            vmin, vmax = limits[name]
#            print(parname,vmin,vmax)
            delta = vmax - vmin
            priors[parname] = stats.uniform(loc=vmin, scale=delta)
        elif parname in vsyst:
            priors[parname] = stats.norm(loc=vsyst[parname], scale=500)
        elif parname == "eta": #what does eta do?
            priors["eta"] = stats.uniform(loc=1., scale=19)#uniform distribution in range [1,19]
        elif parname == "nu": #what does nu do?
            priors["nu"] = stats.uniform(loc=2, scale=20)#uniform distribution in range [2,20]
        elif parname == "sigma":
            priors["sigma"] = stats.uniform(loc=50, scale=300)#obtains the uniform distribution on [loc, loc + scale]. i.e. uniform in range [50,300]
        elif parname == "sigma_gas":
            priors[parname] = stats.uniform(loc=50, scale=100)#uniform between [50,100]km/s
        elif name == "w":
            priors[parname] = stats.uniform(loc=0, scale=1)#weights uniform between 0 and 1
        elif name in linenames:
#             priors[parname] = stats.expon(loc=0, scale=0.5)#favors low values>~0; make even stronger by decreasing scale. 
            priors[parname] = stats.expon(loc=0, scale=0.2)#favors low values>~0; make even stronger by decreasing scale. 
        elif name in ["pred", "pblue"]:
            porder = int(parname.split("_")[1])
            if porder == 0:
                mu, sd = 1 / nssps, 1
                a, b = (0 - mu) / sd, (np.infty - mu) / sd
                priors[parname] = stats.truncnorm(a, b, mu, sd)
            else:
                priors[parname] = stats.norm(0, 0.05)
        else:
            print(f"parameter without prior: {parname}")
    return priors

def log_probability(theta):
    """ Calculates the probability of a model."""
    global priors
    global logp
    lp = np.sum([priors[p].logpdf(x) for p, x in zip(logp.parnames, theta)])
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = logp(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_sampler(outdb, nsteps=5000):
    global logp
    global priors
    ndim = len(logp.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    print(logp.parnames)
#    print(shape(logp.parnames))
#    print(shape(priors))
#    print(ndim)
#    print(nwalkers)
    for i, param in enumerate(logp.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
#    print(shape(pos))    
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
#    print(outdb)
    try:
        pool_size = context.mp_pool_size
    except:
        pool_size = 1
#    print(pool_size)
#    print(shape(backend))
    pool = mp.Pool(pool_size)
#    print(pool)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
#        print(sampler)
        sampler.run_mcmc(pos, nsteps, progress=True)
        
#     resultfile = outdb.replace(".h5", "_all.h5")
#     file=h5py.File(resultfile,"w")
# #    file.create_dataset('sampler',data=sampler)
#     samples = sampler.chain
#     file.create_dataset('samples',data=samples)
#     file.create_dataset('logp',data=logp)
#     file.close()

    return

def weighted_traces(parnames, trace, nssps):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    weights = np.array([trace["w_{}".format(i+1)].data for i in range(
        nssps)])
    wtrace = []
    for param in parnames:
        data = np.array([trace["{}_{}".format(param, i+1)].data
                         for i in range(nssps)])
        t = np.average(data, weights=weights, axis=0)
        wtrace.append(Table([t], names=["{}_weighted".format(param)]))
    return hstack(wtrace)

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
#     print(shape(data))
#     print(size(data))
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, output,
                 skylines=None, dsky=3):
    width_ratios = [w[-1]-w[0] for w in waves]
    fig, axs = plt.subplots(2, len(seds), gridspec_kw={'height_ratios': [2, 1],
                            "width_ratios": width_ratios},
                            figsize=(2 * context.fig_width, 3))
    for i in range(len(waves)):
        sed = seds[i]
        t = np.array([trace[p].data for p in sed.parnames]).T
        n = len(t)
        pmask = np.where(masks[i]==0, True, False)
        wave = waves[i][pmask]
        flux = fluxes[i][pmask]
        fluxerr = fluxerrs[i][pmask]
        models = np.zeros((n, len(wave)))
        y = np.percentile(models, 50, axis=(0,))
        for j in tqdm(range(len(trace)), desc="Generating models "
                                                         "for trace"):
            models[j] = seds[i](t[j])[pmask]
        y = np.percentile(models, 50, axis=(0,))
        yuerr = np.percentile(models, 84, axis=(0,)) - y
        ylerr = y - np.percentile(models, 16, axis=(0,))
        ax0 = fig.add_subplot(axs[0,i])
        ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax0.plot(wave, y, c="tab:orange")
        ax0.xaxis.set_ticklabels([])
        ax0.set_ylabel("Flux")

        ax1 = fig.add_subplot(axs[1,i])
        ax1.errorbar(wave, 100 * (flux - y) / flux, yerr=100 * fluxerr, \
                                                                fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax1.plot(wave, 100 * (flux - y) / flux, c="tab:orange")
        ax1.set_ylabel("Res. (%)")
        ax1.set_xlabel("$\lambda$ (Angstrom)")
        ax1.set_ylim(-5, 5)
        ax1.axhline(y=0, ls="--", c="k")
        # Include sky lines shades
        if skylines is not None:
            for ax in [ax0, ax1]:
                w0, w1 = ax0.get_xlim()
                for skyline in skylines:
                    if (skyline < w0) or (skyline > w1):
                        continue
                    ax.axvspan(skyline - 3, skyline + 3, color="0.9",
                               zorder=-100)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    return

def plot_corner(trace, outroot, title=None, redo=False):
    labels = {"imf": r"$\Gamma_b$", "Z": "[Z/H]", "T": "Age (Gyr)",
              "alphaFe": r"[$\alpha$/Fe]", "NaFe": "[Na/Fe]",
              "Age": "Age (Gyr)", "x1": "$x_1$", "x2": "$x_2$", "Ca": "[Ca/H]",
              "Fe": "[Fe/H]", "Na": "[Na/H]",
              "K": "[K/H]", "C": "[C/H]", "N": "[N/H]",
              "Mg": "[Mg/H]", "Si": "[Si/H]", "Ca": "[Ca/H]", "Ti": "[Ti/H]"}
    title = "" if title is None else title
    output = "{}_corner.png".format(outroot)
    if os.path.exists(output) and not redo:
        return
    N = len(trace.colnames)
    params = trace.colnames
    data = np.stack([trace[p] for p in params]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    title = [title]
    for i, param in enumerate(params):
        parname = param.replace("_weighted", "")
        s = "{0}$={1:.2f}^{{+{2:.2f}}}_{{-{3:.2f}}}$".format(
            labels[parname], v[i], vuerr[i], vlerr[i])
        title.append(s)
    fig, axs = plt.subplots(N, N, figsize=(3.54, 3.5))
    grid = np.array(np.meshgrid(params, params)).reshape(2, -1).T
    for i, (p1, p2) in enumerate(grid):
        p1name = p1.replace("_weighted", "")
        p2name = p2.replace("_weighted", "")
        i1 = params.index(p1)
        i2 = params.index(p2)
        ax = axs[i // N, i % N]
        ax.tick_params(axis="both", which='major',
                       labelsize=4)
        if i // N < i % N:
            ax.set_visible(False)
            continue
        x = data[:,i1]
        if p1 == p2:
            sns.kdeplot(x, shade=True, ax=ax, color="C0")
        else:
            y = data[:, i2]
            sns.kdeplot(x, y, shade=True, ax=ax, cmap="Blues")
            ax.axhline(np.median(y), ls="-", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 16), ls="--", c="k", lw=0.5)
            ax.axhline(np.percentile(y, 84), ls="--", c="k", lw=0.5)
        if i > N * (N - 1) - 1:
            ax.set_xlabel(labels[p1name], size=7)
        else:
            ax.xaxis.set_ticklabels([])
        if i in np.arange(0, N * N, N)[1:]:
            ax.set_ylabel(labels[p2name], size=7)
        else:
            ax.yaxis.set_ticklabels([])
        ax.axvline(np.median(x), ls="-", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 16), ls="--", c="k", lw=0.5)
        ax.axvline(np.percentile(x, 84), ls="--", c="k", lw=0.5)
    plt.text(0.6, 0.7, "\n".join(title), transform=plt.gcf().transFigure,
             size=8)
    plt.subplots_adjust(left=0.12, right=0.995, top=0.98, bottom=0.08,
                        hspace=0.04, wspace=0.04)
    fig.align_ylabels()
    for fmt in ["png", "pdf"]:
        output = "{}_corner.{}".format(outroot, fmt)
        plt.savefig(output, dpi=300)
    plt.close(fig)
    return
    
    
def run_paintbox(galaxy, spec, V0s, dlam=100, nsteps=5000, loglike="normal2",
                 nssps=1, target_res=None):
    """ Run paintbox. """
    global logp, priors
#     target_res = [200, 100] if target_res is None else target_res
    target_res = [180, 100] if target_res is None else target_res
    # List of sky lines to be ignored in the fitting
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836,
                         8919, 9310])
    dsky = 3 # Space around sky lines
    wdir = os.path.join(context.home_dir, f"data/{galaxy}")
    spec2 = os.path.join(wdir, f"{spec}")
#     # Providing template file
#     spec = os.path.join(wdir, f"{spec}.fits")
#    spec = os.path.join(wdir, f"{galaxy}_spec.fits")
    logps = []
#    wranges = [[4000, 6680], [7800, 8900]]
    wranges = [[4040, 7064], [8143, 8838]]#use outermost available lambda
#     waves, fluxes, fluxerrs, masks, seds = [], [], [], [], []
    waves, fluxes, fluxerrs, masks, seds, linenames = [], [], [], [], [], []
    for i, side in enumerate(["blue", "red"]):
        # Locationg where pre-processed models will be stored for paintbox
        sigma = target_res[i]

        # Reading the data
        tab = Table.read(spec2, hdu=i+1)
        #  Normalizing the data to make priors simple
        norm = np.nanmedian(tab["flux"])
        wave = tab["wave"].data
        flux = tab["flux"].data / norm
        fluxerr = tab["fluxerr"].data / norm
        mask = tab["mask"]
#     store = os.path.join(context.home_dir,
#                          f"templates/CvD18_sig{sigma}_{name}_ll{round(wmin)}_{round(wmax)}_{round(velscale)}.fits")
#         store = os.path.join(context.home_dir, "templates",
#                              f"CvD18_sig{sigma}_{side}.fits")
        velscale = sigma / 2
        wmin = wave.min() - 180
        wmax = wave.max() + 50
        store = os.path.join(context.home_dir, "templates",
                             f"CvD18_sig{sigma}_{side}_ll{round(wmin)}_{round(wmax)}_{round(velscale)}.fits")
        print(store)
        if not os.path.exists(store):
            # Compiling the CvD models
            twave = disp2vel([wmin, wmax], velscale)
            CvD18(twave, sigma=sigma, store=store, libpath=context.cvd_dir)

        idx = np.where((wave < wranges[i][0]) | (wave > wranges[i][1]))[0]
        mask[idx] = False
        mask[idx] = 1 	#6.8.21 anja, set to 1 as is bad value

        # Masking all remaining locations where flux is NaN
        mask[np.isnan(flux * fluxerr)] = 1

        # Masking lines from Osterbrock atlas
        for line in skylines:
            idx = np.argwhere((wave >= line - dsky) &
                              (wave <= line + dsky)).ravel()
            mask[idx] = 1.

        # Defining polynomial order
        wmin = wave[mask==0].min()
        wmax = wave[mask==0].max()
        porder = int((wmax - wmin) / dlam)

        # Building paintbox model
#         sed, limits = make_paintbox_model(wave, nssps=nssps, name=side,
#                                   sigma=target_res[i], porder=porder)
        sed, limits, lines = make_paintbox_model(wave, store,
                              nssps=nssps, name=side, sigma=sigma,
                              porder=porder)
# # anja mask edges of flux, as flux drops
# #         plot(wave,flux)
#         mask[0:10]=1
#         mask[-10:]=1
#         plot(wave[mask == 0],flux[mask == 0])
        logp = pb.Normal2LogLike(flux, sed, obserr=fluxerr, mask=mask)
        logps.append(logp)
        waves.append(wave)
        fluxes.append(flux)
        fluxerrs.append(fluxerr)
        masks.append(mask)
        seds.append(sed)
        linenames += lines

    # Make a joint likelihood for all sections
    logp = logps[0]
    for i in range(nssps - 1):
        logp += logps[i+1]

    # Making priors
#     v0 = {"vsyst_blue": 1590, "vsyst_red": 1590}
#     priors = set_priors(logp.parnames, limits, vsyst=v0, nssps=nssps)
    v0 = {"vsyst_blue": V0s[0], "vsyst_red": V0s[1]}
    priors = set_priors(logp.parnames, limits, linenames, vsyst=v0, nssps=nssps)
#     print(logp.parnames)
#     print(limits)

    # Perform fitting
#    dbname = f"{galaxy}_nssps{nssps}_{loglike}_nsteps{nsteps}.h5"
#     dbname = f"{spec}_nssps{nssps}_{loglike}_nsteps{nsteps}.h5"
    pb_dir = f"paintbox_nssps{nssps}_{loglike}_nsteps{nsteps}"
#    print(pb_dir) #paintbox_nssps2_normal2_nsteps100
    if not os.path.exists(pb_dir):
        os.mkdir(pb_dir)
        
    tmp_dir = os.path.join(wdir, pb_dir)
#    print(tmp_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
        
    dbname = os.path.join(pb_dir, spec.replace(".fits", ".h5"))
    print(dbname)
    
    # Run in any directory outside Dropbox to avoid problems
    tmp_db = os.path.join(os.getcwd(), dbname)
#    print(tmp_db)
    #/Users/afeldmei/Magellan/Baade/analysis/paintbox/run_lowmassgal/scripts_v2/paintbox_nssps2_normal2_nsteps5000/NGC4033_1_1arc_bg_noconv.h5
    if os.path.exists(tmp_db):
        os.remove(tmp_db)	#deletes existing result file. dangerous. forces new fit each time

    outdb = os.path.join(wdir, dbname)
    print(outdb)
    if not os.path.exists(outdb):
        print('run mcmc')
        run_sampler(tmp_db, nsteps=nsteps)
        shutil.move(tmp_db, outdb)

    # Post processing of data
    if context.node in context.lai_machines: #not allowing post-processing @LAI
        return
    reader = emcee.backends.HDFBackend(outdb)#, read_only=True
    shape(reader)
    tracedata = reader.get_chain(discard=int(nsteps * 0.9), flat=True, thin=100)	#discard 90% of steps, only every 100. step
    print(logp.parnames)
    nvar=len(logp.parnames)
    print('nvariables',nvar)
    burnin=int(nsteps * 0.5)
    thin=100
    tracedata = reader.get_chain(discard=burnin, flat=True, thin=thin)	#discard 50% of steps, only every 100. step
    print('tracedata shape')
    print(shape(tracedata))
    print(shape(logp.parnames))

    trace = Table(tracedata, names=logp.parnames)
    print(len(trace))
    if nssps > 1:
        ssp_pars = list(limits.keys())
        wtrace = weighted_traces(ssp_pars, trace, nssps)
        trace = hstack([trace, wtrace])
    print((ssp_pars))#['Z', 'Age', 'x1', 'x2', 'C', 'N', 'Na', 'Mg', 'Si', 'Ca', 'Ti', 'Fe', 'K', 'Cr', 'Mn', 'Ba', 'Ni', 'Co', 'Eu', 'Sr', 'V', 'Cu', 'a/Fe']
    print(len(wtrace))
    print(len(trace))
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
#    print(outtab)
#    print(trace)
    make_table(trace, outtab)
    # Plot fit
    outimg = outdb.replace(".h5", "_fit.png")
    plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, outimg,
                 skylines=skylines)
#     print(shape(wave))
#     print(shape(fluxes))
#     print(shape(seds))
#     print(shape(masks))
#     print(max(masks),min(masks))

    # Make corner plot
    # Choose columns for plot
    cols_for_corner = [_ for _ in trace.colnames if _.endswith("weighted")]
    corner_table = trace[cols_for_corner]
    corner_file = outdb.replace(".h5", "_corner") # It will be saved in png/pdf
    plot_corner(corner_table, corner_file, title=galaxy, redo=False)
    
        #https://emcee.readthedocs.io/en/latest/tutorials/monitor/
    
    samples = reader.get_chain(discard=burnin).reshape((-1,nvar))
    print('samples shape')
    print(shape(samples))
    #sampler.chain[:, burnin:, :].reshape((-1, ndim))
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
#     log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
#     log_prior_samples = reader.get_blobs(flat=True)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(tracedata.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
#     print(shape(log_prior_samples))
#  #   print("flat log prior shape: {0}".format(log_prior_samples.shape))
#     samples = np.concatenate((tracedata, log_prob_samples[:, None]), axis=1)
#     all_samples = np.concatenate((tracedata, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1)
#     labels = logp.parnames#list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
#    labels += ["log prob", "log prior"]

    xarr=np.arange(size(log_prob_samples))
#     plot(xarr,log_prob_samples)



if __name__ == "__main__":
#     V0s = {"NGC7144": (1390, 1860), "NGC4033": (1617, 1617)}
    galaxies = ["NGC4033","NGC4387","NGC4458"]
    V0s = {"NGC4033": (1617, 1617), "NGC4387": (540, 540), "NGC4458": (670, 670)}

    galaxies = ["NGC4033"]#,"NGC4387","NGC4458"]
    V0s = {"NGC4033": (1617, 1617)}
    for galaxy in galaxies:
###         specs=['NGC4033_1_1arc']#,'NGC4033_2_outReff16','NGC4033_3_Reff24toReff8','NGC4033_4_Reff8toReff4','NGC4033_5_Reff4toReff2','NGC4033_6_Reff2toReff34','NGC4033_7_Reff34toReff']#,'NGC4033_10_reff8','NGC4033_8_Reff2toReff']
        wdir = os.path.join(context.home_dir, f"data/{galaxy}")
        os.chdir(wdir)
        specs = sorted([_ for _ in os.listdir(".") if _.endswith(
                        "1_1arc_bg_noconv.fits")])
#                        "2_outReff16_bg_noconv.fits")])

        print(specs)
        os.chdir('../../scripts_v2')
        V0 = V0s[galaxy]
        for spec in specs:
#            print(galaxy,spec)
            run_paintbox(galaxy, spec, V0, nssps=2, nsteps=6000)
