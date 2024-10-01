print('\nLoading data...\n')
import uproot
import warnings
import os
import decimal
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import chisquare
import matplotlib as mpl
font = {'family' : 'serif', 'size' : 12 }
mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = 'cm' # Set the math font to Computer Modern
mpl.rcParams['legend.fontsize'] = 1
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
from tqdm import trange
from tqdm import tqdm

# --------------------------------------------------------- #

# plot title label
id_names = ["NCQE"]

# save path for plots
SavePath = 'plots/NCQE/NCQE_2023/15_hits/'

# directory for root files
directory = '2023_prestine/'

# total POT label
total_POT = r'$0.862 \times 10^{20}$ POT'   # 2023 usable data

# depending on which event class, you likely will need to adjust:
# line ~530:      (some numbers between 0 and 20 ns)
#                   fitStart = ...
#                   fitEnd = ...

# --------------------------------------------------------- #


def extract_run_number(file_name):
    return int(file_name.split('_')[0][1:])


# Event selection:

# throughgoing muons
def throughgoing(T1,TMRD1,NV1,B1,CPE1,CCB1,hitZ,hitPE):
    if(T1==0):    # 1 track (1st because this is the strictest selection criteria)
        return False
    if(TMRD1==0): # TankMRDCoinc
        return False
    if(NV1==1):   # NoVeto
        return False
    if(B1==0):    # Brightest
        return False
    if(CPE1<2000 or CPE1>6000):  # 2000 < cluster PE < 6000
        return False
    if(CCB1>0.2 or CCB1<0): # Cluster Charge Balance < 0.2
        return False
    a = 0
    for i in range(len(hitZ)):  # charge barycenter downstream
        a += hitZ[i]*hitPE[i]
    if(a<0):
        return False
    return True

# Basic nu_mu CC inclusive
def CC_inc(T1,TMRD1,NV1,B1,CN1,CPE1):
    if(T1==0):    # 1 track (1st because this is the strictest selection criteria)
        return False
    if(TMRD1==0): # TankMRDCoinc
        return False
    if(NV1==0):   # NoVeto
        return False
    if(B1==0):    # Brightest
        return False
    if(CN1!=0):   # first cluster
        return False
    if(CPE1<200 or CPE1>5000):  # 200 < cluster PE < 5000
        return False
    return True

# NC events
def NC(TMRD1,NV1,NOC1,CN1,CPE1,CH1,CCB1,MRD_yes,FMV_yes):
    if(TMRD1==1): # TankMRDCoinc
        return False
    if(NV1==0):   # NoVeto
        return False
    if(NOC1!=1):  # only cluster
        return False
    if(CN1!=0):   # first cluster (should be the same as above)
        return False
    if(CPE1<20 or CPE1>70):  # 10 < cluster PE < 80
        return False
    if(CH1<14):              # less than 10 clustered hits
        return False
    if(CCB1<=0.3 or CCB1>1):
        return False
    if(MRD_yes==1):     # any MRD hits
        return False
    if(FMV_yes==1):     # any FMV hits
        return False
    return True


file_list = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

sigma_list = []
slopes = []
slopes_error = []

# enable if using 2023 data due to bad BRF
#filtered_file_list = [file for file in file_list if extract_run_number(file) >= 4119]# or extract_run_number(file) == 4111]
#filtered_file_list = [file for file in file_list if extract_run_number(file) != 3790]

print('There are: ', len(file_list), ' files\n')

ct_min = []

counted = 1; CT_list = []; count = 0
for file_name in file_list:

    with uproot.open(directory + file_name) as file:

        print('\nRun: ', file_name, '(', (counted), '/', len(file_list), ')')
        print('------------------------------------------------------------')
    
        Event = file["data"]
        #RN1 = Event["run_number"].array()
        #PF1 = Event["p_file"].array()
        #EN1 = Event["event_number"].array()
        CT1 = Event["cluster_time"].array()
        CT2 = Event["cluster_time_BRF"].array()
        CPE1 = Event["cluster_PE"].array()
        CCB1 = Event["cluster_Qb"].array(library="np")
        NOC1 = Event["number_of_clusters"].array()
        CN1 = Event["cluster_Number"].array()
        CH1 = Event["cluster_Hits"].array()
        B1 = Event["isBrightest"].array()
        TMRD1 = Event["TankMRDCoinc"].array()
        MRD_yes = Event["MRD_activity"].array()
        FMV_yes = Event["FMV_activity"].array()
        T1 = Event["MRD_Track"].array()
        NV1 = Event["NoVeto"].array()
        #POT1 = Event["POT"].array()
        HZ1 = Event['hitZ'].array()
        HPE1 = Event['hitPE'].array()
        HT1 = Event['hitT'].array()
        

        counted += 1

        events_per_run = 0
        for i in trange(len(CT2)):

            #is_thru = throughgoing(T1[i],TMRD1[i],NV1[i],B1[i],CPE1[i],CCB1[i],HZ1[i],HPE1[i])
            #if(is_thru==False):
            #   continue
            
            is_NC = NC(TMRD1[i],NV1[i],NOC1[i],CN1[i],CPE1[i],CH1[i],CCB1[i],MRD_yes[i],FMV_yes[i])
            if(is_NC==False):
                continue

            #is_CC = CC_inc(T1[i],TMRD1[i],NV1[i],B1[i],CN1[i],CPE1[i])
            #if(is_CC==False):
             #  continue

            #if CT1[i] == CT2[i]:          # BRFFirstPeakFit = 0
             #   continue

            #CT_list.append(CT1[i])        # normal clusterTime

            #CT_list.append((CT1[i] - CT2[i]) + np.median(HT1[i]))

            CT_list.append(CT2[i])       # BRF subtracted clusterTime
            
            events_per_run += 1

        print(events_per_run, ' events')

timing_data = [CT_list]
print('\nAfter selection cuts, we have: ', len(CT_list), ' events\n')


# Display timing distribution to find the first bunch
print('Please find the center of the first bunch on the following plot...\n')

dump = [value for value in CT_list if value < 2000]

dump_bins = np.arange(100, 1900, 4)
h_dump, _ = np.histogram(dump, bins=dump_bins)
plt.hist(dump_bins[:-1], bins=dump_bins, weights=h_dump, alpha=0.7, color='darkblue')
plt.xlim([100, 360])
plt.show()

# Ask the user for input based on the plot
bunch_start = float(input("First bunch peak time?   "))

# Close the plot and continue with the script
plt.close('all')

# Continue with the rest of your script
print(f"You entered: {bunch_start}", '\n')


# gaussian function
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# gaussian fit
def fit_gaussian(x, y, initial_mean):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizeWarning)
            params, _ = curve_fit(gaussian, x, y, p0=[max(y), initial_mean, 10], bounds=([max(y)-20, initial_mean-6, 0], [max(y)+20, initial_mean+6, np.inf]), maxfev=5000)
        return params
    except (RuntimeError, ValueError) as e:
        print(f"Fitting error at initial_mean={initial_mean}: {e}")
        return None

# perform fit and create plots
def perform_fit_and_plot(timing_data, id_name, bunch_true_spacing, first_peak, SavePath):
    
    result = []
    bins_TSDt = np.arange(100, 1900, 2)
    h_TSDt, _ = np.histogram(timing_data, bins=bins_TSDt)
    
    # first peak should be somewhere between 200 and 260, search for it
    print('First bunch start set to: ', first_peak)
    
    start_i = first_peak - 12; end_i = first_peak + 5
    
    plt.figure(figsize=(4, 3))
    plt.hist(bins_TSDt[:-1], bins=bins_TSDt, weights=h_TSDt, alpha=0.7, color='darkblue')
    plt.axvline(first_peak, color = 'r', linestyle = 'dashed')
    plt.xlabel('nanoseconds', loc = 'right')
    plt.title(f'First bunch start {id_name}')
    plt.xlim([first_peak - 50,first_peak + 50])
    plt.tight_layout()
    plt.savefig(SavePath + f'FindingFirstBunch_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()
    

    fitResult = []
    print('\nFitting bunch structure...\n')
    for step in trange(int((end_i - start_i) * 10)):#, desc="Fitting steps"):
        current_step = start_i + step * 0.1
        chi2_list = []
        mean_list = []
        sigma_list = []
        for i in range(81):
            start = current_step + i * bunch_true_spacing       # known bunch spacing
            end = start + bunch_true_spacing
            mask = (bins_TSDt[:-1] >= start) & (bins_TSDt[:-1] < end)
            x = bins_TSDt[:-1][mask]
            y = h_TSDt[mask]
            
            params = fit_gaussian(x, y, start + bunch_true_spacing/2)
            if params is None:
                chi2_list.append(np.nan)
                mean_list.append(np.nan)
                sigma_list.append(np.nan)
                continue
            
            y_fit = gaussian(x, *params)
            
            y_sum = np.sum(y)
            y_fit_sum = np.sum(y_fit)
            
            if y_sum == 0 or y_fit_sum == 0:
                chi2_list.append(np.nan)
                mean_list.append(np.nan)
                sigma_list.append(np.nan)
                continue
            
            y = y / y_sum
            y_fit = y_fit / y_fit_sum

            chi2, _ = chisquare(y, y_fit)
            
            chi2_list.append(chi2)             # chi2 vals for each of the 81 bunches
            mean_list.append(params[1])        # list of the bunch means
            sigma_list.append(params[2])       # list of the std dev
        
        if chi2_list:
            fitResult.append([step, mean_list, sigma_list, chi2_list])

    best_fit = min(fitResult, key=lambda x: np.nansum(x[3]))    
    # [0] = step number that provided the best fit
    # [1] = list of mean values for each of the 81 bunches at the best-fitting step
    # [2] = list of std vals
    # [3] = list of chisq vals
    
    result.append(best_fit)
    best_fit_means = best_fit[1]     # bunch centers
    print(best_fit_means)
    

    # Histogram displaying the time distribution
    plt.figure(figsize=(16, 5))
    plt.hist(bins_TSDt[:-1], bins=bins_TSDt, weights=h_TSDt, alpha=0.7, color='tomato', histtype = 'stepfilled')
    plt.hist(bins_TSDt[:-1], bins=bins_TSDt, weights=h_TSDt, alpha=0.7, color='black', histtype = 'step', linewidth = 2)
    plt.text(0.2, 0.9, total_POT, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=20)
    #plt.axvline(750, color = 'blue', linestyle = 'dashed')
    #plt.axvline(1750, color = 'blue', linestyle = 'dashed')
    plt.xlabel('Cluster Time [ns]', fontsize = 16)
    plt.ylabel('Events / 2ns', fontsize = 16)
    plt.title(f'Beam Timing Histogram for {id_name}', fontsize = 24)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    plt.tight_layout()
    plt.savefig(SavePath + f'TimingHist_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()
    

    # Timing distribution with gaussian fits
    plt.figure(figsize=(20, 5))
    plt.hist(bins_TSDt[:-1], bins=bins_TSDt, weights=h_TSDt, alpha=0.7, color='C0')
    for i in range(81):
        start = best_fit[1][i] - bunch_true_spacing/2
        end = start + bunch_true_spacing
        mask = (bins_TSDt[:-1] >= start) & (bins_TSDt[:-1] < end)
        x = bins_TSDt[:-1][mask]
        if len(x) < 3:
            continue
        params = fit_gaussian(x, h_TSDt[mask], start + bunch_true_spacing/2)
        if params is None:
            continue
        y_fit = gaussian(x, *params)
        plt.plot(x, y_fit, 'r-', linewidth=1)
    plt.xlabel('Cluster Time [nanoseconds]')
    plt.ylabel('Counts')
    plt.title(f'Beam Timing with Gaussian Fits for {id_name}')
    plt.tight_layout()
    plt.savefig(SavePath + f'TimingHist_gaussian_fitting_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()

    # Gaussian fits by themselves
    '''
    plt.figure(figsize=(20, 5))
    for i in range(81):
        start = best_fit[1][i] - bunch_true_spacing/2
        end = start + bunch_true_spacing
        mask = (bins_TSDt[:-1] >= start) & (bins_TSDt[:-1] < end)
        x = bins_TSDt[:-1][mask]
        if len(x) < 3:
            continue
        params = fit_gaussian(x, h_TSDt[mask], start + bunch_true_spacing/2)
        if params is None:
            continue
        y_fit = gaussian(x, *params)
        plt.plot(x, y_fit, 'r-', linewidth=1)
    plt.xlabel('Cluster Time [ns]')
    plt.ylabel('Counts')
    plt.title(f'Gaussian Fits for {id_name}')
    plt.tight_layout()
    plt.savefig(SavePath + f'Gaussian_fits_only_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()

    # Gaussian fits with best fit highlights
    plt.figure(figsize=(20, 5))
    plt.hist(bins_TSDt[:-1], bins=bins_TSDt, weights=h_TSDt, alpha=0.7, color='C0')
    for i in range(81):
        start = best_fit[1][i] - bunch_true_spacing/2
        end = start + bunch_true_spacing
        mask = (bins_TSDt[:-1] >= start) & (bins_TSDt[:-1] < end)
        x = bins_TSDt[:-1][mask]
        if len(x) < 3:
            continue
        params = fit_gaussian(x, h_TSDt[mask], start + bunch_true_spacing/2)
        if params is None:
            continue
        y_fit = gaussian(x, *params)
        if(params[2]>8):
            plt.plot(x, y_fit, 'r-', linewidth=1)
        else:
            plt.plot(x, y_fit, 'lime', linewidth=1)
    plt.xlabel('Cluster Time [ns]')
    plt.ylabel('Counts')
    plt.title(f'Gaussian Fits with Best Fit Highlights for {id_name}')
    plt.tight_layout()
    plt.savefig(SavePath + f'Gaussian_fits_with_highlights_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()


    steps = [result[0] for result in fitResult]
    chi2_sums = [np.nansum(result[3]) for result in fitResult]
    chi2_vars = [np.nanvar(result[3]) for result in fitResult]
    # Chisq values for each fit
    plt.figure(figsize=(10, 6))
    plt.scatter(steps, chi2_sums, color='C0', alpha=0.7)
    plt.ylim(0, 50)
    plt.xlabel('Initial Step Position')
    plt.ylabel('Sum of Chi^2')
    plt.title(f'Sum of Chi^2 for Each Step {id_name}')
    plt.savefig(SavePath + f'chisq_sums_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()

    # Variance of Chisq
    plt.figure(figsize=(10, 6))
    plt.scatter(steps, chi2_vars, color='C0', alpha=0.7)
    plt.ylim(0.0001, 0.02)
    plt.xlabel('Initial Step Position')
    plt.ylabel('Variance of Chi^2')
    plt.title(f'Variance of Chi^2 for Each Step {id_name}')
    plt.savefig(SavePath + f'chisq_vars_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()
    '''

    # Interval between bunches
    interval = []
    for i in range(1, len(best_fit[1])):
        interval.append(best_fit[1][i] - best_fit[1][i-1])
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(interval, range=(15, 25), bins=30, alpha=0.7, color='C0')
    plt.axvline(x=bunch_true_spacing, color='r', linestyle='--', linewidth=1, label = 'Prediction = ' + str(bunch_true_spacing) + ' ns')
    plt.xlabel('Fitted Interval (ns)')
    plt.ylabel('Counts')
    plt.title(f'Bunch spacings {id_name}')
    plt.savefig(SavePath + f'bunch_spacings_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()
    '''

    # This was blank for some reason
    interval_diff = []
    for i in range(len(best_fit[1])-1):
        interval_diff.append((best_fit[1][i+1] - best_fit[1][i]) - bunch_true_spacing)
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(interval_diff, range=(-5, 5), bins=30, alpha=0.7, color='C0')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Time Diff (ns)')
    plt.ylabel('Counts')
    plt.title(f'Bunch Spacings to Best Prediction {id_name} (prediction = ' + str(bunch_true_spacing) + ' ns)')
    plt.tight_layout()
    plt.savefig(SavePath + f'spacings_diff_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()
    '''
    
    interval_diff_std = np.std(interval_diff)
    print(f'Interval difference standard deviation for {id_name}: {interval_diff_std}')
    
    # sigma
    sigma_values = best_fit[2]  # sigma
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(sigma_values, bins=20, alpha=0.7, color='C0', range = (0,12))
    plt.xlabel('Bunch Sigma [ns]')
    plt.ylabel('Counts')
    plt.title(f'Bunch widths for {id_name}')
    plt.tight_layout()
    plt.savefig(SavePath + f'bunch_sigma_{id_name}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()
    '''
    
    sigma_mean = np.mean(sigma_values)
    print(f'bunch sigma mean for {id_name}: {sigma_mean}')
    sigma_diff_std = np.std(sigma_values)
    print(f'bunch sigma standard deviation for {id_name}: {sigma_diff_std}')
    
    return result




# Run it

for i in range(len(timing_data)):
    FitResult = perform_fit_and_plot(timing_data[i], id_names[i], 18.936, bunch_start, SavePath)

    # x data of points from 0 to 81
    x_data = np.arange(len(FitResult[0][1]))
    # bunch times
    y_data = FitResult[0][1]
    
    def linear_func(x, a, b):
        return a * x + b
    params, covariance = curve_fit(linear_func, x_data, y_data)
    slope = params[0]
    intercept = params[1]

    # 计算斜率的误差
    slope_error = np.sqrt(covariance[0][0])

    print(f"Bunch fit Slope: {slope}, Slope Error: {slope_error}")

    # for some reason this returns all the bunches
    plt.figure(figsize=(5, 4))
    plt.scatter(x_data, y_data, label='Data', s=10)
    plt.plot(x_data, linear_func(x_data, *params), 'r-', label=f'Fit: y = {slope:.4f} $\pm$ {slope_error:.4f} ns')
    plt.text(0.95, 0.2, total_POT, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=14)
    plt.xlabel('Bunch number')
    plt.ylabel('Bunch Center time [ns]')
    plt.title(f'Linear fit to bunch spacings')
    plt.legend(fontsize = 10)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(SavePath + f'bunch_linear_fit_spacings_{id_names[i]}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
    plt.close()

    slopes.append(slope)
    slopes_error.append(slope_error)

print("The slopes are: ", slopes)
print("The slope errors are: ", slopes_error)

#Take the average of the slopes for slope value is between 18.9 to 19.0:
#SelectedSlopes = [slope for slope in slopes if 18.9 < slope < 19.0]
#SelectedSlopesError = [slopes_error[slopes.index(slope)] for slope in slopes if 18.9 < slope < 19.0]

SelectedSlopes = [slope for slope in slopes if 18.5 < slope < 19.5]
SelectedSlopesError = [slopes_error[slopes.index(slope)] for slope in slopes if 18.5 < slope < 19.5]

slope = np.mean(SelectedSlopes)

#slope = slopes[0]

print("Selected slopes for BRF is: ", slope)
print("The error of the average of the selected slopes is: ", np.sqrt(np.sum([error ** 2 for error in SelectedSlopesError]) / len(SelectedSlopesError)))
print('\n')



# Superimpose bunches
print('Superimposing all bunches...\n')

bins = 40
ranges = (0, 20)

# may need to be modified depending on events
fitStart = 2
fitEnd = 17

steps = []
means = []
sigmas = []


plt.figure(figsize=(10, 6))
for step in tqdm(np.arange(-10, 0, 0.05)):

    decimal_step = Decimal(str(step))  # Convert step to Decimal
    newTime = [(Decimal(str(i)) - decimal_step) % Decimal(slope) for i in timing_data[0]]

    counts, bin_edges, _ = plt.hist(newTime, bins=bins, range=ranges, alpha=0.2, label=f'step={step}')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = (bin_centers >= fitStart) & (bin_centers <= fitEnd)
    bin_centers_fit = bin_centers[mask]
    counts_fit = counts[mask]

    initial_guess = [max(counts_fit), np.mean(bin_centers_fit), np.std(bin_centers_fit)]

    try:
        popt, _ = curve_fit(gaussian, bin_centers_fit, counts_fit, p0=initial_guess)
        amplitude, mean, sigma = popt
        # 保存 step、mean 和 sigma
        steps.append(step)
        means.append(mean)
        sigmas.append(sigma)
    except RuntimeError:
        continue

plt.close()

# 转换为 numpy 数组
steps = np.array(steps)
means = np.array(means)
sigmas = np.array(sigmas)

# 找到最小的 mean 和 sigma 对应的 step
min_mean_step = steps[np.argmin(means)]
min_sigma_step = steps[np.argmin(sigmas)]

'''
plt.figure(figsize=(10, 6))
plt.plot(steps, sigmas, 'r-', label='Sigma')
plt.xlabel('Step')
plt.ylabel('Sigma')
plt.title('Sigma vs Step')
plt.legend()
plt.grid()
plt.ylim(2,10)
plt.savefig(SavePath + f'sigma_vs_step_all_{id_names[0]}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
plt.close()
'''

# in the means, find the index that closest to slope/2, use steps[index] as startTime
diff = 1e9
startTime = 0
sigmaHere = 0
for i, mean in enumerate(means):
    if abs(mean - slope / 2) < diff:
        diff = abs(mean - slope / 2)
        startTime = steps[i]
        sigmaHere = sigmas[i]

print(f'Start time: {startTime} with sigma: {sigmaHere}')

#newTime = [(Decimal(str(i)) - Decimal(startTime)) % Decimal(slope) for i in timing_data[0]]

dummy_list = []
for i in range(len(timing_data[0])):
    #if 205 < timing_data[0][i] < 1740:
    if 190 < timing_data[0][i] < 1760:
    #if 210 < timing_data[0][i] < 602:
        dummy_list.append(timing_data[0][i])

newTime = [(Decimal(str(i)) - Decimal(startTime)) % Decimal(slope) for i in dummy_list]

bins = 20
ranges = (0, 20)
#fitStart = slope/2-6
#fitEnd = slope/2+6

plt.figure(figsize=(9, 6))

counts, bin_edges, _ = plt.hist(newTime, bins=bins, range=ranges, alpha=1, color = 'navy', histtype = 'step')
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
widthz = bin_edges[1] - bin_edges[0]

mask = (bin_centers >= fitStart) & (bin_centers <= fitEnd)
filtered_bin_centers = bin_centers[mask]
filtered_counts = counts[mask]

initial_guess = [max(filtered_counts), np.mean(filtered_bin_centers), np.std(filtered_bin_centers)]

popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_counts, p0=initial_guess)
amplitude, mean, sigma = popt

sigma_error = np.sqrt(pcov[2, 2])

x = np.linspace(fitStart, fitEnd, 1000)
y = gaussian(x, amplitude, mean, sigma)
#plt.plot(x, y, 'r-', label = r'$\mu = {:.3f}\,\mathrm{{ns}},\ \sigma = {:.3f} \pm {:.3f}\,\mathrm{{ns}}$'.format(mean, sigma, sigma_error))
plt.plot(x, y, 'r-')

plt.text(0.95, 0.95, total_POT, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=14)
plt.text(0.95, 0.88, r'$N_{events} = $' + str(len(newTime)), horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=14)
plt.text(0.95, 0.81, r'$\sigma = $' + str(round(sigma,2)) + ' ' + r'$\pm$' + ' ' + str(round(sigma_error,2)) + ' ns', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=14)
plt.text(0.95, 0.74, r'$\mu = $' + str(round(mean,2)) + ' ns', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=14)

#plt.text(0.8, 0.2, 'data after 750 ns', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)

plt.xlabel(f'Timing % {slope:.2f} [ns]')
#plt.ylabel('Events / ' + str(widthz) + 'ns')
plt.ylabel('Events / ns')

ax = plt.gca()
ax.tick_params(axis='both', direction='in')
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))

plt.title('Superimposing all bunches ' + id_names[0])
#plt.title('Superimposing final 53 bunches [750:1750] ns ' + id_names[0])
#plt.legend(loc='lower right', fontsize = 14)
plt.xlim([-1,24])
plt.savefig(SavePath + f'superimposed_bunches_{id_names[0]}.png', dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')
plt.close()

# 输出 mean, sigma 和 sigma 的误差
print(f"Mean: {mean}, Sigma: {sigma}, Sigma Error: {sigma_error}")

print('\ndone\n')
