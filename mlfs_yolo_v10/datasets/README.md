# MLFS-YOLO Datasets

This directory contains datasets used by the MLFS-YOLO model, divided into public datasets and custom datasets.

**Due to privacy and intellectual property considerations, for academic collaboration or access to the complete resources, please contact the author directly.**

## Directory Structure

```
datasets/
├── public_dataset/           # Public datasets (original unprocessed)
│   ├── RDD2022.zip           # Road Damage Detection Dataset 2022
│   └── VOC2012.rar           # PASCAL VOC2012 Dataset
├── partial_dataset/          # Partially provided custom datasets 
    ├── data_preprocessing/   # Data preprocessing scripts and tools
    ├── images/               # Sample images (partially provided)  
    └── labels/               # Annotation data (partially provided)
```

## Public Datasets

The public datasets section contains commonly used object detection benchmark datasets for basic model training and evaluation:

1. **PASCAL VOC2012**:
   - Contains 20 categories of common objects
   - Used for basic model performance evaluation and comparison
   - Download link: [VOC2012 Official Website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

2. **RDD2022** (Road Damage Detection Dataset):
   - Contains images and annotations of various road damages
   - Used to evaluate model performance in specific scenarios
   - Download link: [RDD2022 Official Website](https://github.com/sekilab/RoadDamageDetector)

## Custom Datasets (Partially Provided)

**Important Note:** Due to privacy and intellectual property considerations, only a subset of custom datasets is provided for demonstration purposes. The complete datasets are not included in this repository.

Characteristics of the custom datasets:
- Collected for scenarios optimized for MLFS-YOLO
- Contains special cases with small objects and complex backgrounds
- Enhanced training for frequency domain feature adaptability

### Data Preprocessing

The `data_preprocessing` directory contains scripts for processing and preparing datasets, including:
- Data augmentation tools
- Label conversion tools
- Dataset splitting scripts

### Usage Instructions

1. For public datasets, please extract the corresponding compressed files to their respective directories
2. Use the tools in `data_preprocessing` to process the data
3. Specify the corresponding dataset configuration file in the training script

## Academic Collaboration

To obtain the complete custom datasets for academic research, please contact the author:

**Xu Rao**  
ShenSi Lab, Shenzhen Institute for Advanced Study  
University of Electronic Science and Technology of China  

Email: [raoxu@std.uestc.edu.cn](mailto:raoxu@std.uestc.edu.cn) 

