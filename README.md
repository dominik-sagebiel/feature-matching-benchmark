SIFT vs SuperPoint Image Registration Comparison
=================================================

This project compares SIFT and SuperPoint (with SuperGlue) for image registration using FLANN + RANSAC and SuperGlue matching.

Project Structure
-----------------

myproject/  
├───Images/                # Put your input images here (.png, .jpg, .tif)  
├───Python/                 # Python scripts  
│   ├───20points_onlyFLANNransac.py    # SIFT vs SuperPoint (FLANN+RANSAC)  
│   └───20points_all_PC.py             # SuperPoint + SuperGlue, SIFT + FLANN, SP + FLANN, SIFT + SG comparison  
├───R/                      # R scripts  
│   └───rpautratSuperPoint_IMGwithSizeLimit.R  
├───Repos/                  # External repositories (clone here)  
│   ├───SuperPoint/                     # rpautrat/SuperPoint  
│   └───SuperGluePretrainedNetwork/     # MagicLeap/SuperGlue  
├───Results/                # All outputs (auto-created)  
│   ├───Python/             # Python script results  
│   └───R/                  # R script results  
├───requirements.txt        # Python dependencies  
└───README.md              # This file  

Quick Start
-----------

1. Clone this repository

git clone https://github.com/dominik-sagebiel/feature-matching-benchmark.git       
cd myproject

2. Clone external repositories into Repos/

cd Repos  
git clone https://github.com/rpautrat/SuperPoint.git  
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git  
cd ..  

3. Add your images

Place your test images (.png, .jpg, or .tif) in the Images/ folder.

4. Run the scripts

cd Python  
python 20points_onlyFLANNransac.py  
python 20points_all_PC.py  
