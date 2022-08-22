# MRI Image Quality Metrics (IQM) Tool 
Tool to generate image quality metrics for nifti MRI files. Available metrics:
- PSNR 
- SNR 
- CNR
- SSIM

# Setup

1. Download files 
2. Open the config file and enter the filepath for the image directory [optional]

# Usage: 

### 1. File Input
Run the script 'image_quality_metrics_28072022.py'. There are 3 options avaialable for inputting the location of the nifti files for assessment. If you are reviewing multiple files, 
it is best to supply the directory path containing the participant sub-directories. For the directory search to work correctly, 
each participant should have a folder containing their scans, the number nested subfolders within the participant folder should not exceed 5.
Ideally files are in BIDS format, or BIDS-like, for the script to perform best. 

1. Config file: enter the address of the directory containing the participant folders with their scans 
2. Terminal input for multiple files: input the directory in the terminal using the arguments '--directory' or '-d'
3. Terminal input for a single filepair: input the path for two files, separated by whitespace, in the terminal using the arguments '--filepair' or '-f'

**Note:** if this is the first time running the rating tool, a directory will be created titled 'processed_nifti_files'. The processed MRI scans will be stored here.  

### 2. Select metrics 
Enter 'y' to select the metrics to be calculated. Pressing any other key will mean the metric is not calculated. After selecting 
the metrics, you will be prompted to enter a file name for the output .csv file containing the metrics. 

### 3. File conversions 
To measure certain IQMs some scans will requires image processing such as coregistration and skull-stripping. This processing can be time 
consuming depending on your chosen metrics. When a file is processed, the processed output fires are stored in 'processed_nifti_files'. 
Once the processed files have been created, as long as they remain in the 'processed_nifti_files' folder, there will be no need to rerun 
the processing. 

After the conversion files are made available, the IQMs will be calculated. Again, this can take some time depending on your chosen
metrics.

### 4.  Output
A .csv file containing the metrics will be created using the name provided at the beginning of the script. 


