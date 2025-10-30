# AutoSpot

## Overview

Welcome to **AutoSpot**, a project focused on detecting and segmenting autophagy spots in fluorescence microscopy images. This repository contains the code and methodologies developed during the master's thesis titled **"Combining Traditional Image Analysis and Deep Learning for Enhanced Atg8 Spot Analysis in Autophagy Research."**

The thesis is part of the Master’s degree in Computational Science: Bioscience at the University of Oslo.

![Prompts](spot_segmentation/sam2/plots/sam_prompts.png)

## Repository Structure

This repository is organised into several folders, each containing specific codes and functionalities:

- **create_dataset**: This folder includes scripts for preparing fluorescence microscopy images for analysis. It contains codes to extract bounding boxes and perform initial analyses of the ground truth dataset.

- **peak_detection**: In this section, you will find codes for performing peak detection utilising the 'mountains' and 'SpotSegmentor' algorithms.

- **spot_detection**: This folder features scripts for object detection using the YOLO11 model, enabling the identification of relevant spots within microscopy images.

- **spot_segmentation**: Here, you can access various codes for spot segmentation, employing multiple models including SAM2, micro-SAM, U-Net++, YOLO11, DeepLabV3+, Segformer, and SpotSegmentor.

## Getting Started

To utilize the codes contained within this repository, make sure to check the respective folders for detailed instructions and requirements. Each section is tailored to guide you through the setup and execution of the methods for effective Image Analysis in autophagy research.

Thank you for taking the time to explore AutoSpot. We hope you find this resource helpful in your research on autophagy analysis and image processing techniques.

