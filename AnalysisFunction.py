import numpy as np
import matplotlib.pyplot as pl
import time
import scipy.optimize as op
import emcee
from multiprocessing.pool import ThreadPool
import corner
from getdist import plots, MCSamples, loadMCSamples, types

#Minimize chi2
def find_bestfit(lnlike, par_ml, parnames):
    t1 = time.time()
    ndim = len(par_ml)
    chi2 = lambda *args: -2 * lnlike(*args)
    result = op.minimize(chi2, par_ml)
    if not result['success']:
        result = op.minimize(chi2, par_ml,method='Nelder-Mead',options={'maxiter': 10000})#, args=data
    par_ml = result["x"]
    print('Maximum likelihood result:')
    for i in range(ndim):
        print(parnames[i],' = ',par_ml[i])
    print('chi2min =',result['fun'])
    t2 = time.time()
    print("tempo total: {0:5.3f} seg".format(t2-t1))
    return result

#Run MC
def run_emcee(par_ml, nwalkers, lnprob, ainput, nsteps):
    ndim = len(par_ml)
    pos = [par_ml +1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
    with ThreadPool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=ainput, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    
    accept = sampler.acceptance_fraction
    print('Acceptance fraction:',accept)
    print('Minimum acceptance:',np.amin(accept))
    print('Maximum acceptance:',np.amax(accept))
    
    return sampler

def tira_burnin(sampler, burnin, ndim):
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return samples

def burninthin(sampler, tau):
    taumax = np.amax(tau)
    taumin = np.amin(tau)
    samples = sampler.get_chain(discard=int(2*taumax), thin=int(taumin/2), flat=True)
    print(samples.shape)
    return samples

#MC results
def MC_result(samples, par_ml, parnames, sigma=2):
    ndim = len(par_ml)
    par_mean = np.mean(samples,axis=0)
    par_median = np.percentile(samples, [50], axis=0)[0]

    if sigma==2:
        par_valm = np.percentile(samples, [15.865525393149998], axis=0)[0]
        par_valp = np.percentile(samples, [84.13447460685], axis=0)[0]
        par_valm2 = np.percentile(samples, [2.275013194800002], axis=0)[0]
        par_valp2 = np.percentile(samples, [97.7249868052], axis=0)[0]
        par_sigm = par_mean - par_valm
        par_sigp = par_valp - par_mean
        par_sigm2 = par_mean - par_valm2
        par_sigp2 = par_valp2 - par_mean
        print('MCMC result:')
        for i in range(ndim):
            print("""{0} = {1:5.5f} +{2:5.5f} +{3:5.5f} -{4:5.5f} -{5:5.5f} (median: {6:5.5f}, ml: {7:5.5f})"""\
                  .format(parnames[i],par_mean[i],par_sigp[i],par_sigp2[i],par_sigm[i],par_sigm2[i],par_median[i],par_ml[i]))

    elif sigma==1:
        par_valm = np.percentile(samples, [15.865525393149998], axis=0)[0]
        par_valp = np.percentile(samples, [84.13447460685], axis=0)[0]
        par_sigm = par_mean - par_valm
        par_sigp = par_valp - par_mean
        print('MCMC result:')
        for i in range(ndim):
            print("""{0} = {1:5.5f} +{2:5.5f} -{3:5.5f} (median: {4:5.5f}, ml: {5:5.5f})"""\
                  .format(parnames[i],par_mean[i],par_sigp[i],par_sigm[i],par_median[i],par_ml[i]))

def plot_chains(sampler,par_ml,parlabtex):
    ndim = len(par_ml)
    pl.clf()
    fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].axhline(par_ml[i], color="#888888", lw=2)
        axes[i].set_ylabel(parlabtex[i])
        
    axes[ndim-1].set_xlabel("step number")
    
    fig.tight_layout(h_pad=0.0)
    pl.show()

def plot_cornertriangle(samples,parlabtex,par_ml):
    fig = corner.corner(samples,bins=50,labels=parlabtex,truths=par_ml)
    pl.show()

def tau(sampler, nstep):
    tau = sampler.get_autocorr_time()
    print(tau)
    taumax = np.amax(tau)
    r = nstep/taumax
    print(r)
    if(r>50):
        print('Convergiu! :)')
    else:
        print('Nao convergiu... :(')

    return tau

def plotContours(samples, parnames, parlabels, legends, save, prefix, fill, multiplots=False, print_results=False, size=10):
    gsamples = []
    for i in range(len(samples)):
        gsample = MCSamples(samples=samples[i], names=parnames[i], labels=parlabels[i], ranges={'wm':(0, None)})
        gsample.updateSettings({'countours': [0.682689492137, 0.954499736104, 0.997300203937]})
        gsamples.append(gsample)
    g = plots.getSubplotPlotter(width_inch=10)
    g.triangle_plot(gsamples, parnames[0], filled=fill, legend_labels=legends)
    if save == True:
        g.export('fig/g_'+prefix+'_triangle.pdf')
    pl.show()

    if print_results == True:
        print(types.ResultTable(ncol=1,results=gsamples, paramList=parnames, limit=1).tableTex())
        print(types.ResultTable(ncol=1,results=gsamples, paramList=parnames, limit=2).tableTex())