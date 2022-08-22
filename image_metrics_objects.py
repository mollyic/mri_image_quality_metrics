import os
import signal
from hamcrest import none
import numpy as np
import nibabel as nib
import ants
#SSIM packages
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import crop
from skimage.util import img_as_float
import nibabel as nib
from math import isnan
from mriqc.qc import anatomical
from math import pi, sqrt

def kill_process():
    # view the open applications with 'ps ax' then pipe results into grep, searching for 'itk-snap'
    # exclude the actual grep search using exclusive argument '-v'
    for line in os.popen("ps ax | grep -i fsleyes | grep -v grep"):
        #process ID is the first column (0), isolate this item
        pid = line.split()[0]
        #kill the process by providing the PID, sigkill function forcefully tells comp to terminate program
        os.kill(int(pid), signal.SIGKILL)

def measure_psnr(arr1,arr2): 
    x = np.asarray((nib.load(arr1)).get_fdata())
    y = np.asarray((nib.load(arr2)).get_fdata())
    #computer mean squared error: representation of absolute error 
    mse =  np.mean((x - y) ** 2)
    #computer psnr: ratio between maximum possible signal power and power of distorting noise  
    #square= (x.max())**2
    psnr = 10 * np.log10((x.max())**2/mse)
    #alt formulas
    # maxval = max(x.max(), y.max())
    # psnr2 = 10 * np.log10(maxval) - 10 * np.log10(mse)
    # psnr3 = 20 * np.log10(maxval/np.sqrt(mse))
    return psnr

def antspsnr(img1, img2):
    x = ants.image_read(img1)
    y = ants.image_read(img2)
    def mse(x, y=None):
        if y is None:
            x2 = x ** 2
            return x2.mean()
        else:
            diff2 = (x - y) ** 2
            return diff2.mean()

    antspsnr = 20 * np.log10(x.max()) - 10 * np.log10(mse(x, y))
    return antspsnr


class statistics:
    def __init__(self, d1, d2) -> None:
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = np.mean(d1), np.mean(d2)
        # calculate the effect size
        self.cohensd = (u1 - u2) / s
        self.abseffect = u1 - u2

class fileexts:
    def __init__(self, filename) -> None:
        self.name = filename.split('/')[-1].split('.',1)[0]
        self.filewext = filename.split('/')[-1]
        self.ext = '.' + filename.split('/')[-1].split('.',1)[1]
        self.path = filename
        self.directory = filename.replace(filename.split('/')[-1], '')

#function to remove individual values from nifty arrays for statistics 
class nonzero:
    def __init__(self, nifti) -> None:
        #array to list
        ar_to_list = (np.asarray((nib.load(nifti)).get_fdata())).tolist()
        #empty list for all voxels
        voxel_list = [] 

        for a in range(len(ar_to_list)): #List of lists of lists level
            for b in range (len(ar_to_list[a])): #lists of lists Level
                for c in range(len(ar_to_list[a][b])): #list of items in list 
                    voxel_list.append(ar_to_list[a][b][c]) #Add element to list
        
        #create list for voxels > 1 
        nonzero_voxels=[]
        for voxel in voxel_list:
            if voxel: 
                nonzero_voxels.append(voxel)
                    
        #return a list containing all the non-zero voxels for a nifty 
        self.nonzero_lst = nonzero_voxels

class measure_nr():
    RAYLEIGH_FACTOR = 1.0 / sqrt(2 / (4 - pi))
    def __init__(self, wm = False, gm = False, noise = False):
        self.wm = wm
        self.gm = gm
        self.noise = noise 
        #sd of signal within air mask
        self.air_sd = np.std(nonzero(self.noise).nonzero_lst)
    
        if self.wm:
            #mean of signal within white-matter mask
            self.mu_wm = np.mean(nonzero(self.wm).nonzero_lst)
            self.roi = self.mu_wm
            self.wm_sd = np.std(nonzero(self.wm).nonzero_lst)
        if self.gm:
            #mean of signal within grey-matter mask
            self.mu_gm = np.mean(nonzero(self.gm).nonzero_lst)
            self.roi = self.mu_gm
            self.gm_sd = np.std(nonzero(self.gm).nonzero_lst)
    def rayleigh_ratio(self):
        #ratio to determine if there is a rayleigh distribution (should be less than 1.91)
        return (np.mean(nonzero(self.noise).nonzero_lst))/self.air_sd
        
    def cnr(self):
        return float(abs(self.mu_wm - self.mu_gm) / self.air_sd)   
    def mriqc_cnr(self):
        return anatomical.cnr(self.mu_wm, self.mu_gm, self.air_sd, self.wm_sd, self.gm_sd)
    def snr(self):
        return float(abs(self.roi / self.air_sd))
    def snr_rayleigh(self):
        return float(abs(self.roi / self.air_sd * self.RAYLEIGH_FACTOR))
    def mriqc_snr(self):
        return anatomical.snr(self.roi, self.air_sd)
    def mriqc_snr(self):
        return anatomical.snr_dietrich(self.roi, self.air_sd)

#SSIM 
def mssim(image1, image2, sigma = 1.5, **kwargs):
    if isinstance(image1,(np.ndarray)) == False:
        im1 = np.asarray((nib.load(image1)).get_fdata())
        im2 = np.asarray((nib.load(image2)).get_fdata())
    else:
        im1 = image1
        im2 = image2
    # Define constants 
    # 2nd pop argument is default if there is no value in the dictionary for the first argument
    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")

    #11x11 filter with sigma of 1.5 to match Wang et. al.
    truncate = 3.5
    # set win_size used by crop to match the filter size
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1

    #determine the range of pixel intensities 
    maxpixel = max(np.max(im1), np.max(im2))
    minpixel = min(np.min(im1), np.min(im2))
    data_range = maxpixel - minpixel

    if data_range == 0: 
        return np.NaN

    #Convert to float: ndimage filters need floating point data
    im1 = img_as_float(im1)
    im2 = img_as_float(im2)

    #MU: return the arrays x gaussian weights for luminance for local mu value
    filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
    ux = gaussian_filter(im1, **filter_args)
    uy = gaussian_filter(im2, **filter_args)

    #w(array^2)
    uxx = gaussian_filter(im1 * im1, **filter_args)
    uyy = gaussian_filter(im2 * im2, **filter_args)
    #weighted array x weighted array 
    uxy = gaussian_filter(im1 * im2, **filter_args)
    
    #VARIANCE: contrast(x) = w(xi - ux)^2 
    xsigma = (uxx - ux * ux)
    #VARIANCE: contrast(y) = w(yi - uy)^2
    ysigma = (uyy - uy * uy)
    #COVARIANCE: Σ (xi - µx)(yi - µy) expanded formula 
    sigmaxy = (uxy - (ux * uy))
    #define constants 
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    #Calculate SSIM
    NUM1 = 2 * ux * uy + C1
    NUM2 = 2 * sigmaxy + C2
    DEN1 = ux ** 2 + uy ** 2 + C1
    DEN2 = xsigma + ysigma + C2
    SSIM_MATRIX = (NUM1 * NUM2) / (DEN1 *DEN2)

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2
    #compute mean ssim from array, cropping edges and using float64 for accuracy.
    mssim = crop(SSIM_MATRIX, pad).mean(dtype=np.float64)
    return mssim

#function assess SSIM in 'radiologist view': calculates ssim iteratively by viewing each 3 axes for a certain point
def radiologist_ssim(img1, img2):
    #load the image in as an array 
    ref = np.asarray((nib.load(img1)).get_fdata())
    mot = np.asarray((nib.load(img2)).get_fdata())
    mssim_u =[]

    sag_range = ref.shape[0]
    cor_range = ref.shape[1]
    tra_range = ref.shape[2]
    maxslice = max(sag_range, cor_range, tra_range)
    #slice by slice ssim 
    for i in range(0, maxslice):
        ssimmeans = []
        try:
            sag_ref = ref[i, :, :]
            sag_mot = mot[i, :, :]
            iqa_sag = mssim(sag_ref, sag_mot, data_range=sag_mot.max() - sag_mot.min())
            ssimmeans.append(iqa_sag)
        except:
            pass
        #coronal
        try:
            coronal_ref = ref[:, i, :]
            coronal_mot = mot[:, i, :]
            iqa_cor = mssim(coronal_ref, coronal_mot, data_range=coronal_mot.max() - coronal_mot.min())
            ssimmeans.append(iqa_cor)
        except:
            pass
        #transverse
        try:
            tra_ref = ref[:, :, :i]
            tra_mot = mot[:, :, :i]
            iqa_tra = mssim(tra_ref, tra_mot, data_range=tra_mot.max() - tra_mot.min())
            ssimmeans.append(iqa_tra)
        except:
            pass
        
        filter_ssim = list(filter(lambda x: isnan(x) == False, ssimmeans))
        ssim_u = np.mean(filter_ssim)
        mssim_u.append(ssim_u)

    #filter_ssim = list(filter(lambda x: isnan(x) == False, mssim_u))
    return np.mean(filter_ssim)


#function to transform images into relative axe s
def convert_axes(scan):

    #load the image in as an array 
    nifti = np.asarray((nib.load(scan)).get_fdata())
    #determine the length of each axis 
    sag_range = nifti.shape[0]
    cor_range = nifti.shape[1]
    tra_range = nifti.shape[2]
    axis_list = [sag_range, cor_range, tra_range]
    
    #sag is default orientation: immediately add array to list 
    axs_arr =[nifti]

    for coord, niiaxis in enumerate(axis_list[1:3]):
    #create the slices for each axis 
        slice_list = []
        for i in range(0, niiaxis):
            #coronal
            if coord == 0:
                slice = nifti[:, i, :]
            #transverse
            if coord == 1:
                slice = nifti[:, :, i]
            slice_list.append(slice)
        axs_arr.append(np.asarray(slice_list))
    
    return axs_arr


def allaxesssim(im1, im2):
    ref_arrays = convert_axes(im1)
    mot_arrays = convert_axes(im2)
    meanssim = []
    for i in range(0,3):
        iqa_test = mssim(ref_arrays[i], mot_arrays[i], data_range=mot_arrays[i].max() - mot_arrays[i].min(), gaussian_weights=True)
        meanssim.append(iqa_test)
    return np.mean(meanssim)







#OLD WITH FUNCTION

#function to remove individual values from nifty arrays for statistics 
# def nonzero_list(nifty):
#     #array to list
#     ar_to_list = (np.asarray((nib.load(nifty)).get_fdata())).tolist()
    
#     #empty list for all voxels
#     voxel_list = [] 

#     #search through the array and insert each of the values into a new list 
#     #array is 3 dimensional, so a variable is found at the 3rd level 
#     for a in range(len(ar_to_list)): #List of lists of lists level
#         for b in range (len(ar_to_list[a])): #lists of lists Level
#             for c in range(len(ar_to_list[a][b])): #list of items in list 
#                 voxel_list.append(ar_to_list[a][b][c]) #Add element to list
    
#     #create list for voxels > 1 
#     nonzero_voxels=[]
#     for voxel in voxel_list:
#         if voxel != 0.0: 
#             nonzero_voxels.append(voxel)
#     #return a list containing all the non-zero voxels for a nifty 
#     return len(nonzero_voxels)

#     #contrast to noise ratio
# def cnr(wm, gm, air):
#     #mean of signal within white-matter mask
#     mean_wm = np.mean(nonzero_list(wm))
#     #mean of signal within grey-matter mask
#     mean_gm = np.mean(nonzero_list(gm))
#     #mean of signal within air mask
#     sd_air = np.std(nonzero_list(air))
#     #formula for cnr 
#     return float(abs(mean_wm - mean_gm) / sd_air)

# #contrast to noise ratio
# def cnr(wm, gm, air):
#     #mean of signal within white-matter mask
#     mean_wm = np.mean(nonzero(wm).nonzero_lst)
#     #mean of signal within grey-matter mask
#     mean_gm = np.mean(nonzero(gm).nonzero_lst)
#     #mean of signal within air mask
#     sd_air = np.std(nonzero(air).nonzero_lst)
#     #formula for cnr 
#     return float(abs(mean_wm - mean_gm) / sd_air)


# #compute snr, only use one tissue type (gm or wm())
# def snr(roi, air):
#     #mean of signal within white-matter mask
#     mean_roi = np.mean(nonzero_list(roi))
#     #mean of signal within air mask
#     sd_air = np.std(nonzero_list(air))
#     #formula for cnr 
#     return float(abs(mean_roi / sd_air))

#checking voxel length function
#n='/home/unimelb.edu.au/mollyi/Documents/Projects/Repos/image_quality_rating_tool/processed_nifti_files/NC244/ventricles-mask_bet_sub-NC244_ses-20190715_acq-nomotion_run-01_T1w.nii.gz'
# def vox_list(nifty):
#     #array to list
#     ar_to_list = (np.asarray((nib.load(nifty)).get_fdata())).tolist()
#     voxel_list = [] 

#     for a in range(len(ar_to_list)): #List of lists of lists level
#         for b in range (len(ar_to_list[a])): #lists of lists Level
#             for c in range(len(ar_to_list[a][b])): #list of items in list 
#                 voxel_list.append(ar_to_list[a][b][c]) #Add element to list
        
#     #create list for voxels > 1 
#     nonzero_voxels=[]
#     for voxel in voxel_list:
#         if voxel != 0.0: 
#             nonzero_voxels.append(voxel)
#     #return a list containing all the non-zero voxels for a nifty 
#     print(len(nonzero_voxels))
