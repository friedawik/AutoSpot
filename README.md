# This repository is under construction and will be up to date within the firsst week of November, 2025.

# AutoSpot

Welcome to **AutoSpot**, a project focused on detecting and segmenting Atg8 spots in fluorescence microscopy images for autophagy research. This repository contains the code and methodologies developed during the master's thesis titled:

 **"Combining Traditional Image Analysis and Deep Learning for Enhanced Atg8 Spot Analysis in Autophagy Research."**

The thesis is part of the Master’s degree in Computational Science: Bioscience at the University of Oslo.

![segmentation](spot_segmentation/plots/MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x1_y4.png)

## Repository Structure

This repository is organised into several folders, each containing specific codes and functionalities:

- **create_dataset**: This folder includes scripts for preparing fluorescence microscopy images for analysis. It contains codes to divide images into patches, extract bounding boxes, and perform initial analyses of the ground truth dataset.

- **peak_detection**: In this section, you will find codes for performing peak detection utilising the 'mountains' and 'SpotSegmentor' algorithms.

- **spot_detection**: This folder features scripts for object detection using the YOLO11 model, enabling the identification of relevant spots within microscopy images.

- **spot_segmentation**: Here, you can access various codes for spot segmentation, employing multiple models including SAM2, micro-SAM, U-Net++, YOLO11, DeepLabV3+, Segformer, and SpotSegmentor.

## Getting Started

To use the codes contained within this repository, make sure to check the respective folders for detailed instructions and requirements. Each section is tailored to guide you through the setup and execution of the methods for effective Image Analysis in autophagy research.

Thank you for taking the time to explore AutoSpot. We hope you find this resource helpful in your research on autophagy and Atg8 dynamics.

