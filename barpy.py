# Created by Tobias Geron in Spring 2025 at the University of Toronto


### Imports ###
import numpy as np
import photutils.isophote as phot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.image as mpimg
from tqdm.notebook import tqdm
import scipy.stats as stats



### General functions ###

def circdiff(angleA, angleB, units = 'deg'):
    '''
    Calculate difference for two angles, accounting for that 360 deg and 1 deg are near each other.
    '''

    if units == 'rad': # if input in rad, convert to deg
        angleA = angleA/np.pi*180
        angleB = angleB/np.pi*180
        
    a = angleA - angleB
    a = (a + 180) % 360 - 180

    if units == 'rad': #if input in rad, convert output back to rad
        a = a/180*np.pi
    return a




### Ellipse fitting ###



def fit_ellipses(img, sma0 = np.nan, eps0 = np.nan, pa0 = np.nan, x0 = np.nan, y0 = np.nan, config = None):
    '''
    Guidelines for image:
    Try to centre image on centre of the galaxy. Also, try to fill most of the image with the galaxy, but allow for some empty space near the edges.
    TODO: paralellise mode == 3
    '''
    
    if config == None:
        config = fit_ellipses_config()

    # Just fit ellipses once with initial conditions
    if config.mode == 1:

        assert ~np.isnan(sma0) and ~np.isnan(pa0) and ~np.isnan(eps0), "For config.mode==1, we need initial conditions for sma, pa and eps. If these are unknown, try config.mode = 3 or 4."
        
        isolist, geometry = fit_ellipses_internal(img, sma0, eps0, pa0, x0 = x0, y0 = y0, stepsize = config.stepsize, linear = config.linear, 
                              n_iter = 0, verbose = config.verbose, converge_criteria = config.converge_criteria)

        return isolist, geometry


    
    # Fit ellipses using initial conditions, but let it converge
    elif config.mode == 2:

        assert ~np.isnan(sma0) and ~np.isnan(pa0) and ~np.isnan(eps0), "For config.mode==2, we need initial conditions for sma, pa and eps. If these are unknown, try config.mode = 3 or 4."

        isolist, geometry = fit_ellipses_internal(img, sma0, eps0, pa0, x0 = x0, y0 = y0, stepsize = config.stepsize, linear = config.linear, 
                              n_iter = config.n_iter_converge, verbose = config.verbose)

        
        return isolist, geometry
        

    
    # Randomise initial conditions (eps and pa). Take average of results and try again.
    elif config.mode == 3:

        assert ~np.isnan(sma0), "For config.mode==3, we need initial condition for sma. If this ais unknown, try config.mode = 4."

        
        isolists = []
        geometries = []

        for i in range(config.n_iter):
            eps0 = np.random.uniform(0,1)
            pa0 = np.random.uniform(0,180)


            if config.verbose > 0:
                print(f'Round {i}....')

            try:
                isolist, geometry = fit_ellipses_internal(img, sma0, eps0, pa0, x0 = x0, y0 = y0, stepsize = config.stepsize, linear = config.linear, 
                              n_iter = config.n_iter_converge, verbose = config.verbose)
            except:
                if config.verbose > 0:
                    print(f'No solution found. Continuing.')
                    print(f'')
                continue


            isolists.append(isolist)
            geometries.append(geometry)


            if config.verbose > 0:
                print(f'')


        """
        epss = []
        pas = []

        for isolist in isolists:
            iso = isolist.get_closest(sma0) # Get them at sma0. Alternatively: do median? Do median around sma0?
            epss.append(iso.eps)
            pas.append(iso.pa)

        eps0 = np.mean(epss)
        pa0 = stats.circmean(pa0)/np.pi*180

        eps0_std = np.std(epss)
        pa0_std = stats.circstd(pas)


        if config.verbose > 0:
            print(f'Final round with optimal values...')
            
        isolist, geometry = fit_ellipses_internal(img, sma0, eps0, pa0, x0 = x0, y0 = y0, stepsize = config.stepsize, linear = config.linear, 
                          n_iter = config.n_iter_converge, verbose = config.verbose)


        return isolist, geometry
        """

        return isolists, geometries

        




    # Randomise initial conditions (sma, pa, eps), save results
    elif config.mode == 4:
        isolists = []
        geometries = []

        for i in range(config.n_iter):
            eps0 = np.random.uniform(0,1)
            pa0 = np.random.uniform(0,180)
            sma0 = round(np.random.uniform(img.shape[0]/10,img.shape[0]/3)  / config.stepsize) * config.stepsize 

            if config.verbose > 0:
                print(f'Round {i}....')

            try:
                isolist, geometry = fit_ellipses_internal(img, sma0, eps0, pa0, x0 = x0, y0 = y0, stepsize = config.stepsize, linear = config.linear, 
                              n_iter = config.n_iter_converge, verbose = config.verbose)
            except:
                if config.verbose > 0:
                    print(f'No solution found. Continuing.')
                    print(f'')
                continue


            isolists.append(isolist)
            geometries.append(geometry)


            if config.verbose > 0:
                print(f'')


        return isolists, geometries

            
        
    

class fit_ellipses_config:
    
    def __init__(self):
        self.stepsize = 5
        self.linear = True
        self.n_iter_converge = 5
        self.converge_criteria = [2,2,0.01,0.01]

        self.mode = 2
        self.verbose = 1


        self.n_iter = 5
    

    
    

def fit_ellipses_internal(img, sma0, eps0, pa0, x0 = np.nan, y0 = np.nan, stepsize = 5, linear = True, n_iter = 5, verbose = 1, converge_criteria = [2,2,0.01,0.01]):
    '''
    For the PA's to make sense, we assume images are rotated NWSE (well, not yet).

    img should be a 2D array

    PA0 should be in deg. EoN. (NOT TRUE, SEE BELOW)

    The position angle (in radians) of the semimajor axis in relation to the positive x axis of the image array 
    (rotating towards the positive y axis). Position angles are defined in the range. Avoid using as starting 
    position angle of 0., since the fit algorithm may not work properly. 

    TODO: pix PAs orientation

    linear: (bool) Whether to increase stepsize linearly, or geometrically
    stepsize: if linear == True, then sma_(n+1) = sma_n + stepsize), elif linear == False, then sma_(n+1) = sma_n * (1. + stepsize)

    About sma0:
    The starting value for the semimajor axis length (pixels). This value must not be the minimum or maximum semimajor axis length,
    but something in between. The algorithm can’t start from the very center of the galaxy image because the modelling of elliptical 
    isophotes on that region is poor and it will diverge very easily if not tied to other previously fit isophotes. It can’t start 
    from the maximum value either because the maximum is not known beforehand, depending on signal-to-noise. The sma0 value should 
    be selected such that the corresponding isophote has a good signal-to-noise ratio and a clearly defined geometry.
    '''

    img = np.array(img)

    pa0 = pa0/180*np.pi

    if np.isnan(x0):
        x0 = img.shape[1]/2
    if np.isnan(y0):
        y0 = img.shape[0]/2


    if verbose > 0:
        print(f'Starting conditions: x0 = {x0:.4f}, y0 = {y0:.4f}, sma0 = {sma0:.4f}, eps0 = {eps0:.4f}, pa0 = {pa0/np.pi*180:.4f}')
        


    converged = False
    for i in range(n_iter):
        # Step 1: Don't fix centre, use initial guesses
        geometry = phot.EllipseGeometry(x0=x0, y0=y0, sma = sma0, eps=eps0, pa=pa0, fix_center = False)   
        ellipse = phot.Ellipse(img, geometry)
        isolist = ellipse.fit_image(linear=linear, step = stepsize)
    
        # Step 1.5: Get new guesses
        x0_new = np.median(isolist.x0)
        y0_new = np.median(isolist.y0)
        iso = isolist.get_closest(sma0) # Get them at sma0. Alternatively: do median? Do median around sma0?
        eps0_new = iso.eps
        pa0_new = iso.pa


        # Check if it has converged
        if np.abs(x0-x0_new) < converge_criteria[0] and np.abs(y0-y0_new) < converge_criteria[1] and np.abs(eps0-eps0_new) < converge_criteria[2] and np.abs(pa0-pa0_new) < converge_criteria[3]:
            converged = True

        x0 = x0_new
        y0 = y0_new
        eps0 = eps0_new
        pa0 = pa0_new

        if verbose > 0:
            print(f'Iteration {i}: x0 = {x0:.4f}, y0 = {y0:.4f}, sma0 = {sma0:.4f}, eps0 = {eps0:.4f}, pa0 = {pa0/np.pi*180:.4f}')
        
        if converged:
            break
    

    # Step 2: Fix centre, use converged guesses        
    geometry = phot.EllipseGeometry(x0=x0, y0=y0, sma = sma0, eps=eps0, pa=pa0, fix_center = True)   
    ellipse = phot.Ellipse(img, geometry)
    isolist = ellipse.fit_image(linear=linear, step = stepsize)

    return isolist, geometry




def plot_ellipse_results(isolist, vline = np.nan):
    if type(isolist) == list:
        multiple = True
    else:
        multiple = False

    plt.figure(figsize = (4,6))

    plt.subplot(3,1,1)

    if multiple:
        for iso in isolist:
            plt.plot(iso.sma, iso.intens, c = 'k', alpha = 0.5)
    else:
        plt.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=4)

    if ~np.isnan(vline):
        plt.axvline(vline, ls = '--', c = 'k')
        
    plt.ylabel('Intensity')


    plt.subplot(3,1,2)

    if multiple:
        for iso in isolist:
            plt.plot(iso.sma, iso.eps, c = 'k', alpha = 0.5)
    else:
        plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4)

    if ~np.isnan(vline):
        plt.axvline(vline, ls = '--', c = 'k')
        
    plt.ylim(0,1)
    plt.ylabel('Ellipticity')


    plt.subplot(3,1,3)

    if multiple:
        for iso in isolist:
            plt.plot(iso.sma, iso.pa/np.pi*180, c = 'k', alpha = 0.5)
    else:
        plt.errorbar(isolist.sma, isolist.pa/np.pi*180, yerr=isolist.pa_err, fmt='o', markersize=4)

    if ~np.isnan(vline):
        plt.axvline(vline, ls = '--', c = 'k')
        
    plt.ylim(0, 180)
    plt.xlabel('Semimajor Axis Length [pix]')
    plt.ylabel('PA [deg]')

    plt.tight_layout()
    plt.show()



def plot_images(img, isolist, n_ellipses = 10, bar_sma = np.nan):

    model_image = phot.build_ellipse_model(img.shape, isolist)
    residual = img - model_image
    

    # Visualise results
    plt.figure(figsize = (5,5))

    plt.subplot(2,2,1)
    plt.imshow(img, origin='lower')
    plt.title('Data')

    plt.subplot(2,2,2)
    plt.imshow(img, origin='lower')
    plt.title('Data')
    
    smas = np.linspace(np.percentile(isolist.sma,5), np.percentile(isolist.sma,95), n_ellipses)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        plt.plot(x, y, color='white')


    if ~np.isnan(bar_sma):
        iso = isolist.get_closest(bar_sma)
        x, y, = iso.sampled_coordinates()
        plt.plot(x, y, color='red')

    plt.subplot(2,2,3)
    plt.imshow(model_image, origin='lower')
    plt.title('Ellipse Model')

    plt.subplot(2,2,4)
    plt.imshow(residual, origin='lower')
    plt.title('Residual')

    plt.tight_layout()
    plt.show()




class AveragedIsophoteList:

    def __init__(self, sma, pa, pa_err, eps, ellip_err, intens, int_err, n_averaged):
        self.sma = sma
        self.pa = pa
        self.pa_err = pa_err
        self.eps = eps
        self.ellip_err = ellip_err
        self.intens = intens
        self.int_err = int_err
        self.n_averaged = n_averaged


class SmoothenedIsophoteList:

    def __init__(self, sma, pa, pa_err, eps, ellip_err, intens, int_err):
        self.sma = sma
        self.pa = pa
        self.pa_err = pa_err
        self.eps = eps
        self.ellip_err = ellip_err
        self.intens = intens
        self.int_err = int_err
    


def average_isolists(isolists, min_isolists = None):
    # Find all possible smas
    if min_isolists == None: 
        min_isolists = len(isolists) / 2 # Need at least half
        
    smas = sorted(set(np.concatenate([isolist.sma for isolist in isolists])))#[1:] #Remove the 0
    isos = []
    for sma in smas:
    
        # Find the correct isophote in each isolist
        matching_isophotes = []
        for isolist in isolists:
            idx = np.where(isolist.sma == sma)[0]
            for ix in idx:
                matching_isophotes.append(isolist[ix])
    
    
        isos.append(matching_isophotes)
    
    
    
    eps = np.array([np.mean([j.eps for j in isos[i]]) for i in range(len(isos))])
    ellip_err = np.array([np.sqrt(sum([j.ellip_err**2 for j in isos[i]])) for i in range(len(isos))])
    pa = np.array([stats.circmean([j.pa for j in isos[i]]) for i in range(len(isos))])
    pa_err = np.array([np.sqrt(sum([j.pa_err**2 for j in isos[i]])) for i in range(len(isos))])
    intens = np.array([np.mean([j.intens for j in isos[i]]) for i in range(len(isos))])
    int_err = np.array([np.sqrt(sum([j.int_err**2 for j in isos[i]])) for i in range(len(isos))])
    n_averaged = np.array([len([j for j in isos[i]]) for i in range(len(isos))])

    avg_isolist = AveragedIsophoteList(smas,pa,pa_err,eps,ellip_err,intens,int_err,n_averaged)
    return avg_isolist


def calculate_reduced_chi2(img, isolist):
    '''
    With given isolist, calculate reduced chi2
    '''

    model_image = phot.build_ellipse_model(img.shape, isolist)
    residual = img - model_image

    return sum((residual**2 / img).flatten()) / len(img.flatten())




def select_best_isolist(isolists, img):
    reduced_chi2s = []

    for isolist in isolists:
        rchi2 = calculate_reduced_chi2(img, isolist)
        reduced_chi2s.append(rchi2)


    idx = np.argmin(reduced_chi2s)
    return isolists[idx]
    
        


def smooth_isolist(isolist, window = 5):
    '''
    Apply a sliding window over sequence and take median

    Window is in sma units
    '''

    if window in [0,1]:
        return isolist

    pa_smooth = []
    eps_smooth = []
    int_smooth = [] 

    for i in range(len(isolist.sma)):
        sma = isolist.sma[i]
        idx = np.where( np.abs(sma - isolist.sma) <= window )[0]

        pa_smooth.append(stats.circmean(isolist.pa[idx]))
        eps_smooth.append(np.mean(isolist.eps[idx]))
        int_smooth.append(np.mean(isolist.intens[idx]))


    smooth_isolist = SmoothenedIsophoteList(isolist.sma,np.array(pa_smooth), np.array([np.nan] * len(isolist.sma)), 
                                            np.array(eps_smooth), np.array([np.nan] * len(isolist.sma)), 
                                            np.array(int_smooth), np.array([np.nan] * len(isolist.sma)))
    
    return smooth_isolist
    




"""
def find_bar_in_isophotes(isolist, sma_min = 6, eps_threshold = 0.25, eps_drop = 0.1, pa_change = 10, pa_variation = 40):
    # First attempt at finding bar in isolists.
    # Curently I have a minimum bar lenght. Maybe I should smooth instead? 
    #Maybe smooth the eps to find eps_max? To avoid weird jumps?
    # TODO: not implemented smooth increase in eps in bar region yet!

    '''
    pa_change and pa_variation in degs. 
    
    '''
    

    
    # Check if there are any elliptical structures
    inds = np.where( (isolist.sma > sma_min) & (isolist.eps > eps_threshold))[0]
    if len(inds) == 0:
        return np.nan, np.nan, np.nan

    # Define temp bar to be at eps_max
    eps_max = np.max(isolist.eps[inds])
    idx = np.where(isolist.eps == eps_max)[0][0]
    
    bar_sma = isolist.sma[idx]
    bar_pa = isolist.pa[idx]
    bar_eps = isolist.eps[idx]

    # Ellipticity drop check
    # We require the ellipticity to drop by at least 0.1 from the bar’s maximum ellipticity in the outer disk
    # So if this has at least one, then it's fine. But maybe I should smooth and check on smoothened? I want to avoid irregular bumps that would trigger this.
    idxs = np.where( ((bar_eps - isolist.eps) > eps_drop) & (bar_sma < isolist.sma))[0]
    if len(idxs) == 0:
        return np.nan, np.nan, np.nan

    # PA change check
    #and the PA to change by at least 10° from the associated bar PA in the outer disk
    idxs = np.where( (np.abs(circdiff(bar_pa, isolist.pa, units = 'rad')) >pa_change/180*np.pi) & (bar_sma < isolist.sma))[0]
    if len(idxs) == 0:
        return np.nan, np.nan, np.nan


    # Implement stable PA check! Check in bar region that there is no giant PA change
    idxs = np.where( (bar_sma > isolist.sma) & (isolist.sma > sma_min) & (np.abs(circdiff(bar_pa, isolist.pa, units = 'rad')) > pa_variation/180*np.pi ))[0]
    if len(idxs) > 0: # Maybe say that max 10% of ellipses in bar region can have this. To avoid one strange ellipse throwing this off
        return np.nan, np.nan, np.nan


    return bar_sma, bar_eps, bar_pa/np.pi*180
"""


def find_bar_in_isophotes(isolist, sma_min = 6, eps_threshold = 0.25, eps_drop = 0.1, eps_max_flex = 0.1, pa_change = 10, pa_variation = 40, plot = False, img = np.array([])):
    # Second attempt at finding bar in isolists.
    # Curently I have a minimum bar length. Maybe I should smooth instead? 
    #Maybe smooth the eps to find eps_max? To avoid weird jumps?
    # TODO: not implemented smooth increase in eps in bar region yet!

    '''
    pa_change and pa_variation in degs. 
    '''


    # Check if there are any elliptical structures
    inds = np.where( (isolist.sma > sma_min) & (isolist.eps > eps_threshold))[0]
    if len(inds) == 0:
        return np.nan, np.nan, np.nan


    eps_max = np.max(isolist.eps[inds])

    possible_bar_end = []
    for i in range(len(isolist.sma)):
        sma_temp = isolist.sma[i]
        pa_temp = isolist.pa[i]
        eps_temp = isolist.eps[i]
        
        # Check if ellipse is elliptical enough
        if not np.abs(eps_temp-eps_max) < eps_max_flex:
            continue



        # Ellipticity drop check
        # We require the ellipticity to drop by at least eps_drop from the bar’s maximum ellipticity in the outer disk
        # So if this has at least one, then it's fine. But maybe I should smooth and check on smoothened? I want to avoid irregular bumps that would trigger this.
        idxs = np.where( ((eps_temp - isolist.eps) > eps_drop) & (sma_temp < isolist.sma))[0]
        if len(idxs) == 0:
            continue

         
    
        # PA change check
        #and the PA to change by at least 10° from the associated bar PA in the outer disk
        idxs = np.where( (np.abs(circdiff(pa_temp, isolist.pa, units = 'rad')) >pa_change/180*np.pi) & (sma_temp < isolist.sma))[0]
        if len(idxs) == 0:
            continue



        # Implement stable PA check! Check in bar region that there is no giant PA change
        idxs = np.where( (sma_temp > isolist.sma) & (isolist.sma > sma_min) & (np.abs(circdiff(pa_temp, isolist.pa, units = 'rad')) > pa_variation/180*np.pi ))[0]
        if len(idxs) > 0: # Maybe say that max 10% of ellipses in bar region can have this. To avoid one strange ellipse throwing this off
            continue


        # If the current sma survives all checkes, it is a possible bar end
        possible_bar_end.append(i)


    # No bars found
    if len(possible_bar_end) == 0:
        return np.nan, np.nan, np.nan


    # Find largest bar-end
    idx = np.max(possible_bar_end)

    bar_sma = isolist.sma[idx]
    bar_pa = isolist.pa[idx]
    bar_eps = isolist.eps[idx]


    if plot:
        fig = plt.figure(figsize=(10, 6))
        outer_gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.05)
        
        # Left column: the ellispe fits
        left_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0], hspace=0.3)
        ax1 = fig.add_subplot(left_gs[0])
        ax2 = fig.add_subplot(left_gs[1])
        ax3 = fig.add_subplot(left_gs[2])


        ax1.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=4)
        ax1.axvline(bar_sma, ls = '--', c = 'k')
        ax1.set_ylabel('Intensity')

        ax2.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4)
        ax2.axvline(bar_sma, ls = '--', c = 'k')
        ax2.set_ylim(0,1)
        ax2.set_ylabel('Ellipticity')

        ax3.errorbar(isolist.sma, isolist.pa/np.pi*180, yerr=isolist.pa_err, fmt='o', markersize=4)
        ax3.axvline(bar_sma, ls = '--', c = 'k')
        ax3.set_ylim(0, 180)
        ax3.set_xlabel('Semimajor Axis Length [pix]')
        ax3.set_ylabel('PA [deg]')


        
        # Right column: the images
        right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], hspace=0.1)
        ax4 = fig.add_subplot(right_gs[0])
        ax5 = fig.add_subplot(right_gs[1])
        
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])
        
        if img.shape != (0,):
            ax4.imshow(img, origin='lower')    
            ax5.imshow(img, origin='lower')
            
                        
            if ~np.isnan(bar_sma):
                iso = isolist.get_closest(bar_sma)

                # Plot ellipse
                x, y, = iso.sampled_coordinates()
                plt.plot(x, y, color='red')

                # Plot line
                sma = iso.sma
                x0 = iso.x0
                y0 = iso.y0
                pa = iso.pa  # in radians
                dx = sma * np.cos(pa)
                dy = sma * np.sin(pa)
                x_line = [x0 - dx, x0 + dx]
                y_line = [y0 - dy, y0 + dy]
            
                plt.plot(x_line, y_line, color='red', linestyle='-', alpha = 0.5)
            
    
        plt.show()

    return bar_sma, bar_eps, bar_pa/np.pi*180

    

    

def find_disc_in_isophotes(isolist, sma_min = np.nan, buffer = 1.2, plot = False, img =  np.array([])):
    '''
    Find the eps (inclination) and PA of the disc.
    Take mean/median of isophotes outside of bar region. (times a buffer?)

    Take mean of PAs with circmean.

    sma_min can be length of the bar.
    '''
    if np.isnan(sma_min):
        sma_min = 0

    idxs_disc = np.where(isolist.sma > sma_min*buffer)[0]

    disc_eps = np.mean(isolist.eps[idxs_disc])
    disc_pa = stats.circmean(isolist.pa[idxs_disc])

    disc_eps_err = np.std(isolist.eps[idxs_disc])
    disc_pa_err = stats.circstd(isolist.pa[idxs_disc])

    disc_sma = np.mean(isolist.sma[idxs_disc])



    if plot:
        fig = plt.figure(figsize=(10, 6))
        outer_gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.05)
        
        # Left column: the ellispe fits
        left_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0], hspace=0.3)
        ax1 = fig.add_subplot(left_gs[0])
        ax2 = fig.add_subplot(left_gs[1])
        ax3 = fig.add_subplot(left_gs[2])


        ax1.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o', markersize=4)
        ax1.axvline(sma_min, ls = '--', c = 'k')
        ax1.axvspan(0,sma_min*buffer, color = 'k', alpha = 0.3)
        ax1.set_ylabel('Intensity')

        ax2.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4)
        ax2.axvline(sma_min, ls = '--', c = 'k')
        ax2.axvspan(0,sma_min*buffer, color = 'k', alpha = 0.3)
        ax2.set_ylim(0,1)
        ax2.set_ylabel('Ellipticity')

        ax3.errorbar(isolist.sma, isolist.pa/np.pi*180, yerr=isolist.pa_err, fmt='o', markersize=4)
        ax3.axvline(sma_min, ls = '--', c = 'k')
        ax3.axvspan(0,sma_min*buffer, color = 'k', alpha = 0.3)
        ax3.set_ylim(0, 180)
        ax3.set_xlabel('Semimajor Axis Length [pix]')
        ax3.set_ylabel('PA [deg]')


        
        # Right column: the images
        right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], hspace=0.1)
        ax4 = fig.add_subplot(right_gs[0])
        ax5 = fig.add_subplot(right_gs[1])
        
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])
        
        if img.shape != (0,):
            ax4.imshow(img, origin='lower')    
            ax5.imshow(img, origin='lower')
            
                        
            if ~np.isnan(sma_min):
                iso = isolist.get_closest(disc_sma)

                # Plot ellipse
                x, y, = iso.sampled_coordinates()
                plt.plot(x, y, color='red')

                # Plot line
                sma = iso.sma
                x0 = iso.x0
                y0 = iso.y0
                pa = iso.pa  # in radians
                dx = sma * np.cos(pa)
                dy = sma * np.sin(pa)
                x_line = [x0 - dx, x0 + dx]
                y_line = [y0 - dy, y0 + dy]
            
                plt.plot(x_line, y_line, color='red', linestyle='-', alpha = 0.5)
            
    
        plt.show()

    
    return disc_eps, disc_pa/np.pi*180



def eps2inc(eps, q0 = np.nan):
    """
    eps_disc can trivially be converted to inc_disc using:

    cos(i) = 1 - eps = b/a      - (1)
    
    where i is the inclination, epsilon is the ellipticity, a is the minor axis, and b is the major axis. However, this assumes a thin disc. Alternatively, one can use: 
    
    cos^2(i) = (q^2 - q_0^2)/(1 - q_0^2)      - (2)
    where q the axis ratio (q = b/a), and q_0 the intrinsic oblateness of the galaxy. Typical values are q_0 approx 0.2 - 0.25.
    
    Here, we use Eq (1) to calculate the inclination. If q0 is defined, we use Eq (2) instead.
    """

    
    if np.isnan(q0):
       return np.arccos(1-eps)/np.pi*180

    else:
        q = 1 - eps
        return np.arccos(np.sqrt((q**2 - q0**2)/(1 - q0**2)))/np.pi*180
        
        
    
def ellipse_grid(img, sma0, n_pa0 = 10, n_eps0 = 10, plot = False):
    '''
    Try a grid of eps0 and pa0, see how bar properties change
    '''

    df = []
    lst_pa0 = np.linspace(0, 180, n_pa0)
    lst_eps0 = np.linspace(0, 1, n_eps0)

    for pa0 in tqdm(lst_pa0):
        for eps0 in lst_eps0:
            try:
                isolist, _ = fit_ellipses(img, sma0 = sma0, eps0 = eps0, pa0 = pa0, verbose = 0)
                bar_sma, bar_eps, bar_pa = find_bar_in_isophotes(isolist)
                new_row = {'pa0' : pa0, 'eps0' : eps0, 'bar_sma' : bar_sma, 'bar_eps' : bar_eps, 'bar_pa' : bar_pa}
            except:
                new_row = {'pa0' : pa0, 'eps0' : eps0, 'bar_sma' : np.nan, 'bar_eps' : np.nan, 'bar_pa' : np.nan}
            df.append(new_row)

    df = pd.DataFrame(df)


    if plot:
        plt.figure(figsize = (15,5))
                
        plt.subplot(1,3,1)
        plt.scatter(df['eps0'],df['pa0'],c=df['bar_sma'])
        plt.colorbar()
        plt.title('bar_sma')
        plt.xlabel('eps0')
        plt.ylabel('pa0')
        
        plt.subplot(1,3,2)
        plt.scatter(df['eps0'],df['pa0'],c=df['bar_eps'], vmin = 0, vmax = 1)
        plt.colorbar()
        plt.title('bar_eps')
        plt.xlabel('eps0')
        plt.ylabel('pa0')
        
        plt.subplot(1,3,3)
        plt.scatter(df['eps0'],df['pa0'],c=df['bar_pa'], vmin = 0, vmax = 180)
        plt.colorbar()
        plt.title('bar_pa')
        plt.xlabel('eps0')
        plt.ylabel('pa0')
        
        plt.tight_layout()
        plt.show()


        
    return df