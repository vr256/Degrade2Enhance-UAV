# Degrade2Enhance-UAV

## Overview

Degrade2Enhance-UAV is a customizable stochastic degradation pipeline designed for enhancing UAV-captured images. 

This repository provides a framework for generating a structured dataset of paired images, which includes ground truth images and their corresponding synthetically degraded versions.

The tool allows easy parameter tuning and real-time visualization through an interactive widget, making it ideal for training generative networks aimed at artifact removal.

## Dataset
[Degrade2Enhance-UAV Dataset](https://www.kaggle.com/datasets/vr256x/degrade2enhance-uav)  
It contains clear, ground-truth images primarily from UAVs, along with some from other types of aircraft.  

## Usage

### Setup
- Clone the repository and install required dependencies.
- Configure the DATAPATH environment variable to point to your data directory or enter absolute path to the dataset in `main.ipynb`.
### Running the Pipeline
- Adjust degradation parameters using the GUI provided in the Jupyter notebook.
- Apply the degradation pipeline to your dataset and generate the structured dataset.


## Example

![gif not found](example.gif)
