SIFT vs SuperPoint Image Registration Comparison
=================================================

This project compares SIFT and SuperPoint (with SuperGlue) for image registration using FLANN + RANSAC and SuperGlue matching.

Project Structure
-----------------

feature-matching-benchmark/     
├───Images/                # Put your input images here (.png, .jpg, .tif)  
├───Python/                 # Python scripts  
│   ├───20points_onlyFLANNransac.py    # SIFT vs SuperPoint (FLANN+RANSAC)  
│   └───20points_all_PC.py             # SuperPoint + SuperGlue, SIFT + FLANN, SP + FLANN, SIFT + SG comparison  
│   └───Resize_image.py                # Resizes image and saves it in Images  
│   └───Superpoint_timing.py           # Measures Superpoint inference time and img preprocessing time  
├───R/                      # R scripts  
│   └───Superpoint_timing_noResizing.R        # Runs Superpoint with resized image from Resize_image.py  
│   └───Superpoint_timing_withResizing.R      # Resizes Image and runs Superpoint   
├───Repos/                  # External repositories (clone here)  
│   ├───SuperPoint/                     # rpautrat/SuperPoint  
│   └───SuperGluePretrainedNetwork/     # MagicLeap/SuperGlue  
├───Results/                # All outputs (auto-created)  
│   ├───Python/             # Python script results  
│   └───R/                  # R script results  
└───README.md              # This file  

Quick Start
-----------

1. Clone this repository

git clone https://github.com/dominik-sagebiel/feature-matching-benchmark.git       
cd feature-matching-benchmark  

2. Clone external repositories into Repos/

cd Repos  
git clone https://github.com/rpautrat/SuperPoint.git  
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git  
cd ..  

3. Add your images

Place your test images (.png, .jpg, or .tif) in the Images/ folder or use examples images

4. Run the scripts  
