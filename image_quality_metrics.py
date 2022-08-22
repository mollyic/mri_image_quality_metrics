import configparser
import glob

from importlib_metadata import files
from image_metrics_objects import fileexts, mssim, radiologist_ssim, allaxesssim, statistics, measure_nr, measure_psnr, antspsnr
import subprocess
import csv
import os
import argparse
from collections import defaultdict
import scipy.stats as stats
import numpy as np
from os import path
import nibabel as nib
from skimage.metrics import structural_similarity as ssimsci
from datetime import date
import tqdm

#optional use of arguments to input files 
parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, const=None, help='Provide path to directory containing nii files divided into subdirectories e.g. BIDS format.')
parser.add_argument('--filepair', '-f', nargs="+", default=None, help='Provide path for two files, separated by whitespace.')
args = parser.parse_args()

#empty list of participants (only one if single processing occuring)
participants =[]
#create a dictionary to store the files paths for nii images  
subjectdict = defaultdict(dict)

#if no args provided, use directory path provided in config folder
multi = True 
if not (args.directory or args.filepair):
   config = configparser.ConfigParser()
   config.read('iqm_code_git/image_qual_metrics.ini')
   participants = glob.glob(config['settings']['folders'])
   sessions = glob.glob(config['settings']['folders']+'/*/')
   print(f'\n{len(participants)} participants for review\n')

else:
   # input argument is a directory of images to process 
   if args.directory:
      print(f'Directory: {args.directory}')
      participants = glob.glob(args.directory+'/*/')
      sessions = glob.glob(args.directory+'/*/*/')
      if len(participants) == 0: 
         participants = glob.glob(args.directory+'/')
      print(f'\n{len(participants)} participants for review\n')

   #input argument is a filepair (participants variable = list of 2 files)  
   if args.filepair:
      participants.append(args.filepair)
      sessions.append(args.filepair)
      multi = False

#check the measures to calculate 
psnr = True if input(f"Calculate PSNR? (y/n)") == 'y' else False
cnr = True if input(f"Calculate CNR? (y/n)") == 'y' else False
snr = True if input(f"Calculate SNR? (y/n)") == 'y' else False
ssim = True if input(f"Calculate SSIM? (y/n)") == 'y' else False

# #option to verify with outside modules 
# if psnr:
#    antsverify = True if input(f"Compute PSNR using ANTS? (y/n)") == 'y' else False
# if ssim: 
#    scikitverify = True if input(f"Compute SSIM using scikit? (y/n)") == 'y' else False
if cnr or snr: 
   rayleigh = True if input(f"Calculate Raleigh distribution for air mask? (y/n)") == 'y' else False

csvfilename = input("Name of output csv file (note: date autoadded): ")

#create a folder for the processed images
#check if output directory exists
if path.exists("processed_nifti_files/") == False:
    os.mkdir("processed_nifti_files/")
    print("\n\nDirectory 'processed_nifti_files/' created! Processed images stored here. \n\n")

#empty lists for psnr t-test statistics 
final_metrics =[]
tt_orig = []
tt_betcoreg = []

#COUNTER for testing (ending early (break if counter =2))
counter = 0


# identifier for high resolution images 
highresid = '-highres'
lowresid = '-lowres'

#search the directory or or list of lists in the case of the single participant arg (only 1 list for the participant) 
print("\nCHECKING FOR PROCESSED SCANS\nIf files aren't located, they will be created from the original scans")
for participant in tqdm.tqdm(participants):
   for folder in sessions:
      counter +=1

      #create a list of unprocessed images 
      orig = []
      
      #create file pairings from the folders - simulate what would happen if only two files were selected 
      if multi == True: 
         #look for nii.gz and .nii files 
         types = [folder +'**/*.nii.gz', folder + '**/*.nii']
         
         #create list of original unprocessed files 
         for nifti in types: 
            files_list = glob.glob(nifti, recursive = True)
            orig += files_list

      #used in the case of single subject argument: all input files assumed to be originals   
      else:
         for nifti in folder:
            orig.append(nifti) 

      #check that the subjects are the same
      ids = []
      for file in orig:
         subjnum = file.split('sub-')[-1].split('_',1)[0]
         sesh = file.split('ses-')[-1].split('_')[0]
         subjid = subjnum+'_'+sesh
         ids.append(subjid)
      uniqueids = list(set(ids))
      #create a subject id: ids for processing from different subjects are concatenated 
      if len(uniqueids) != 1:
         for i in uniqueids:
            num = i.split('sub-')[1].split('_',1)[0]
            subjnum += num+'_'
            seshs = i.split('ses-')[1].split('_',1)[0]
            sesh += seshs+'_'
         dicsubject =  subjnum + '_' + sesh
      else: 
         #dic id
         dicsubject = ids[0]

      #used to identify the string that differentiates high res and low res files 
      #check the first file for the acquisition id
      if highresid not in orig[0] and lowresid not in orig[0]:
         highresid = (input('String in filename use to identify high resolution image? (e.g. nomotion): ')).lower()
         lowresid = (input('String in filename use to identify low resolution image? (e.g. motion): ')).lower()

      highres = []
      lowres = []
      #ensure that the first entry of the dictionary is the highres image (e.g. not motion affected) for the PSNR calculation
      orig.sort()
      for file in orig:
         if highresid in file.lower():
            highres.append(file)
         if lowresid in file.lower():
            lowres.append(file)
      
      data = defaultdict(dict)
      for i, file in enumerate(highres): 
         if i == 0: 
            data[fileexts(file).name] = {'type':'ref', 'orig':file, 'coreg':'', 'bet_orig':'', 'bet_coreg': '', 'wm-label':'', 'gm-label':'', 'air-label':'', 'csf-label':'', 'ventricles-label':''}
         else: 
            data[fileexts(file).name] = {'type':'highres','orig':file, 'coreg':'', 'bet_orig':'', 'bet_coreg': '', 'wm-label':'', 'gm-label':'', 'air-label':'', 'csf-label':'', 'ventricles-label':''}
      for i, file in enumerate(lowres):
         data[fileexts(file).name] = {'type':'lowres','orig':file, 'coreg':'', 'bet_orig':'', 'bet_coreg': '', 'wm-label':'', 'gm-label':'', 'air-label':'', 'csf-label':'', 'ventricles-label':''}

      #data[fileexts(highres).name] = {defaultdict(dict)}

      #create a directory for the processed images 
      proc_files = []
      direc = f"processed_nifti_files/{subjnum}/{sesh}/"
      if path.exists(direc) == False:
         os.makedirs(f"processed_nifti_files/{subjnum}/{sesh}")
         print(f"\n\nDirectory '{subjnum}/{sesh}' created! Processed images stored here. \n\n")


      #input items into the dictionary if files already exist for directory entries 
      #elif multi == True: 
      else:
         proc_files = glob.glob(direc+'*')  
         for check_convert in proc_files:
            #check for labels files 
            if 'label' in check_convert:
               if 'bet' in check_convert:
                  ref = fileexts(check_convert).name.split('label_bet_',1)[1]
               else:
                  ref = fileexts(check_convert).name.split('label_',1)[1]
               #files created from white matter probability mask
               if 'gm-' in check_convert:
                  data[ref]['gm-label'] = check_convert
               #files created from grey matter probability mask
               if 'wm-' in check_convert:
                  data[ref]['wm-label'] = check_convert            
               #files created from airmask 
               if 'air-' in check_convert:
                  data[ref]['air-label'] = check_convert
               if 'csf-' in check_convert:
                  data[ref]['csf-label'] = check_convert
               if 'ventricles-' in check_convert:
                  data[ref]['ventricles-label'] = check_convert
            #check for skull stripped files 
            elif 'bet' in check_convert and 'fast' not in check_convert and 'mask' not in check_convert:
               #REF- is a naming convention used to refer the image used as the reference space 
               if 'REF-' in check_convert:
                  #ref = check_convert.split('REF-')[-1].split('.',1)[0]
                  ref = 'sub-'+ (check_convert.split('sub-',3)[-2].split('_REF-')[0])
                  data[ref]['bet_coreg'] = check_convert
               #if the bet filename does not have ref, it is a skull-stripped version of the original image 
               else:
                  ref = fileexts(check_convert).name.replace('bet_', '').split('.',1)[0]
                  data[ref]['bet_orig'] = check_convert
            #check for coregistered files 
            elif "flirt" in str(check_convert) and "fast" not in str(check_convert) and 'mask' not in check_convert:
               ref = 'sub-'+ (check_convert.split('sub-',3)[-2].split('_REF-')[0])
               #ref = str(check_convert.split('REF-')[-1].split('.',1)[0])
               data[ref]['coreg'] = check_convert
      
      #Optional file convert for FLIRT
      if psnr or ssim:
         #search through the dictionary keys based off the file names for each participant folder 
         #select the reference scan
         for nifti in data.keys():
            if data[nifti]['type'] == 'ref':
               refscan = data[nifti]['orig']
         
         for nifti in data.keys():
            if data[nifti]['type'] == 'ref':
               continue
            #add nii.gz file extension as fsl converts to nii.gz not nii
            output = direc + 'flirt_'+ nifti + '_REF-' + fileexts(refscan).name + '.nii.gz'                
            
            #Check if coregistered scan already exists if not, run the conversion
            if not data[nifti]['coreg']:
               print(f"\nCoregistering scan: {nifti} \nReference: {fileexts(refscan).name}")
               proc = subprocess.Popen(['flirt', "-in", data[nifti]['orig'], "-ref", refscan, "-out", output], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
               proc.wait()
               data[nifti]['coreg'] = output
      
      #Optional file convert for BET
      if psnr or cnr or snr:
         for n, nifti in enumerate(data.keys()): 

            #list of bets to convert: cnr and snr don't require coregistration, hence filtering 
            betsconv = []
            #Check if skull stripped scan of original already exists
            if not data[nifti]['bet_orig']:
               betsconv.append('orig')
            
            #Check if skull stripped scan of flirt-coreg already exists (CHECK)
            if psnr and not data[nifti]['bet_coreg']:
               if data[nifti]['type'] != 'ref':
                  betsconv.append('coreg')

            for nifti2bet in betsconv:
               betinput = fileexts(data[nifti][nifti2bet]).filewext
               betoutput= 'bet_'+fileexts(data[nifti][nifti2bet]).name + '.nii.gz'
               
               #create dictionary keys 
               if nifti2bet == 'orig':
                  ref = 'bet_orig'
               if nifti2bet == 'coreg':
                  ref = 'bet_coreg'

               proc = subprocess.Popen(['bet',  data[nifti][nifti2bet],  direc + betoutput, '-R'])
               print(f"\nSkull-stripping scan: {betinput}")
               proc.wait()
               data[nifti][ref]= direc + betoutput
      
      #Optional file convert with fast 
      if cnr or snr:
         #search to see if masks already exist
         for n, nifti in enumerate(data.keys()):
            
            #objects for necessary files (bet and originals)
            originalnii = fileexts(data[nifti]['orig'])
            betnii = fileexts(data[nifti]['bet_orig'])
      
            #Check if wm mask of original already exists
            if not data[nifti]['wm-label'] or not data[nifti]['gm-label'] or not data[nifti]['air-label'] or not data[nifti]['csf-label']:
               #create a file name for the new fast processed files 
               fastorig = direc +'fast_airmasking_'+ originalnii.name
               fastbet = direc + 'fast_tissuemasking_'+betnii.name 
               
               #paths for the label files to be created by fast 
               airlabel = direc+'air-label_'+nifti+'.nii.gz'
               gmlabel = direc+'gm-label_'+nifti+'.nii.gz'
               wmlabel = direc+'wm-label_'+nifti+'.nii.gz'
               csflabel = direc+'csf-label_'+nifti+'.nii.gz'

               #RUN FAST 
               #non-betted for airmask
               proc = subprocess.Popen(['fast', '-o', fastorig, '-R', '0.75', originalnii.path])
               proc.wait()
               #betted for tissues
               proc = subprocess.Popen(['fast', '-o', fastbet, '-R', '0.75', betnii.path])
               proc.wait()
               
               #CREATE AIR MASK and LABEL
               #create a binary mask (values of 0 (air), values >=1 (tissue)), and invert the mask by subtracting 1 (tissue = 0, air = -1)
               # and taking the absolute value 
               proc = subprocess.Popen(['fslmaths', fastorig + '_mixeltype.nii.gz', '-bin', '-fillh', '-sub', '1', '-abs', airlabel.replace('label','mask')])
               proc.wait()
               #apply the mask to the original image to create the air label
               proc = subprocess.Popen(['fslmaths', originalnii.path, '-mas', airlabel.replace('label','mask'), airlabel])
               proc.wait()

               #CREATE GM WM MASKS and CSF using BETTED images 
               #mask for gm = pve_1
               proc = subprocess.Popen(['fslmaths', fastbet + '_pve_1', '-thr', '0.99', '-uthr', '1.01', gmlabel.replace('label','mask')])
               proc.wait()

               #mask for wm = pve_2
               proc = subprocess.Popen(['fslmaths', fastbet + '_pve_2', '-thr', '0.99', '-uthr', '1.01', wmlabel.replace('label','mask')])
               proc.wait()
               
               #mask for csf = pve_0
               proc = subprocess.Popen(['fslmaths', fastbet + '_pve_0', '-thr', '0.99', '-uthr', '1.01', csflabel.replace('label','mask')])
               proc.wait()

               #apply the mask to the betted image to create the tissue labels
               #gm
               proc = subprocess.Popen(['fslmaths', betnii.path, '-mas', gmlabel.replace('label','mask'), gmlabel])
               proc.wait()
               #wm
               proc = subprocess.Popen(['fslmaths', betnii.path, '-mas', wmlabel.replace('label','mask'), wmlabel])
               proc.wait()
               #csf
               proc = subprocess.Popen(['fslmaths', betnii.path, '-mas', csflabel.replace('label','mask'), csflabel])
               proc.wait()

               data[nifti]['air-label']= airlabel
               data[nifti]['gm-label']= gmlabel
               data[nifti]['wm-label']= wmlabel
               data[nifti]['csf-label']= csflabel

      subjectdict[dicsubject] = data

      # if counter == 3:
      #    break
   
#iterate through the dictionary of nii files and perform the IQAs 
print("\nCALCULATING CHOSEN IQMS")

#Results dictionary 
#subject: {scan: {type: '', value1:'', value2:''}
subjresultsdic = defaultdict(dict)

#tqdm for progress bar 
#subject level 
for subj in tqdm.tqdm(subjectdict.keys()):
   resultsdic = defaultdict(dict)
   files_list = list(subjectdict[subj]) 
   #iterate through files 
   for i in subjectdict[subj]:
      if subjectdict[subj][i]['type'] == 'ref':
         refscan = subjectdict[subj][i]['orig']
         betrefscan = subjectdict[subj][i]['bet_orig']

   files_list.remove(fileexts(refscan).name)
   for scan in files_list:
      #subject: {scan: {type: '', value1:'', value2:''}
      scantype = subjectdict[subj][scan]['type']
      resultsdic[scan] = defaultdict(dict)
      resultsdic[scan]['type'] = scantype
      #calculate PSNR 
      if psnr:
         #psnr for raw images (good-good), (bad-bad)
         resultsdic[scan]['raw_psnr'] = round(float(measure_psnr(refscan, subjectdict[subj][scan]['orig'])), 3)
         #psnr for coreg images (good-good), (bad-bad)
         resultsdic[scan]['coreg_psnr'] = round(float(measure_psnr(refscan, subjectdict[subj][scan]['coreg'])), 3)
         #PSNR for skull stripped images
         resultsdic[scan]['bet_psnr'] = round(float(measure_psnr(betrefscan, subjectdict[subj][scan]['bet_orig'])), 3)
         #PSNR for coregistered skull stripped images
         resultsdic[scan]['betcoreg_psnr'] = round(float(measure_psnr(betrefscan, subjectdict[subj][scan]['bet_coreg'])), 3)
         
         #lists for psnr t-statistic 
         tt_orig.append(resultsdic[scan]['raw_psnr'])
         tt_betcoreg.append(resultsdic[scan]['betcoreg_psnr']) 
   
      if cnr:
         #air mask 
         resultsdic[scan]['cnr'] = round(float(measure_nr(subjectdict[subj][scan]['wm-label'], subjectdict[subj][scan]['gm-label'], subjectdict[subj][scan]['air-label']).cnr()), 3)
         
      if snr:
         #wm snr w AIR MASK
         resultsdic[scan]['snr_wm'] = round(float(measure_nr(wm = subjectdict[subj][scan]['wm-label'], noise = subjectdict[subj][scan]['air-label']).snr()), 3)
         #wm snr w AIR MASK with Raleigh constant
         resultsdic[scan]['raysnr_wm'] = round(float(measure_nr(wm = subjectdict[subj][scan]['wm-label'], noise = subjectdict[subj][scan]['air-label']).snr_rayleigh()), 3) 
         
         #gm snr w AIR MASK 
         resultsdic[scan]['snr_gm'] = round(float(measure_nr(wm = subjectdict[subj][scan]['gm-label'], noise = subjectdict[subj][scan]['air-label']).snr()), 3)
         #gm snr w AIR MASK with Raleigh constant
         resultsdic[scan]['raysnr_gm'] = round(float(measure_nr(gm = subjectdict[subj][scan]['gm-label'], noise = subjectdict[subj][scan]['air-label']).snr_rayleigh()), 3) 

      if ssim:
         #3D SSIM with 3D Gaussian kernel  
         #original motion to no motion
         resultsdic[scan]['raw_ssim']= round(float(mssim(refscan, subjectdict[subj][scan]['orig'])), 3)
         #coregistered motion to no motion
         resultsdic[scan]['coreg_ssim'] = round(float(mssim(refscan, subjectdict[subj][scan]['coreg'])), 3)
         
         # #RADIOLOGISTS VIEW SSIM   
         # #original motion to no motion
         # resultsdic[scan]['raw_ssim_radio']= round(float(radiologist_ssim(refscan, subjectdict[subj][scan]['orig'])), 3)
         # #coregistered motion to no motion
         # resultsdic[scan]['coreg_ssim_radio'] = round(float(radiologist_ssim(refscan, subjectdict[subj][scan]['coreg'])), 3)
         
         # #BY AXIS VIEW
         # resultsdic[scan]['raw_ssim_axes']= round(float(allaxesssim(refscan, subjectdict[subj][scan]['orig'])), 3)
         # #coregistered motion to no motion
         # resultsdic[scan]['coreg_ssim_axes'] = round(float(allaxesssim(refscan, subjectdict[subj][scan]['coreg'])), 3)
   
   subjresultsdic[subj]= resultsdic     

#stats processing
if psnr:
   print(f"Cohen's d for PSNR scores (original, processed scans): {statistics(tt_orig, tt_betcoreg).cohensd}")
   print(f"Absolute effect size for PSNR scores (original, processed scans): {statistics(tt_orig, tt_betcoreg).abseffect}")
   t= stats.ttest_rel(tt_orig, tt_betcoreg)
   print(f'T-statistic paired sample t-test (PSNR)(orig-coreg+bet):{round(t[0],3)}, (p= {round(t[1],5)})')

today = date.today()
datecsv = today.strftime("%d%m%Y")

filenamecsv= f'results/{csvfilename}_{datecsv}.csv'

fields = ['subject', 'file']
subjectlist = list(subjresultsdic.keys())
for p in subjresultsdic[subjectlist[0]].keys():
    for j in subjresultsdic[subjectlist[0]][p].keys():
        fields.append(j)
    break

with open(filenamecsv, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for sub in subjresultsdic:
        for file in subjresultsdic[sub]:
            row = {'subject': sub}
            row={field: subjresultsdic[sub][file].get(field) or file for field in fields}
            row.update({'subject': sub})
            writer.writerow(row)






# #if multiple files are parsed, a new csv file will automatically be created  
# if multi:
#    with open(csvfile, 'w') as file:
#       write = csv.writer(file)
#       write.writerow(headers)
#       write.writerows(final_metrics)


# #if single files have been input the csv will be appended with new values instead of creating a new file
# else:
#    if not os.path.exists(csvfile):
#       with open(csvfile, 'w') as file:
#          write = csv.writer(file)
#          write.writerow(headers)
#          write.writerows(final_metrics)

#    else:
#       with open(csvfile, 'a') as file:
#          write = csv.writer(file)
#          write.writerows(final_metrics)

