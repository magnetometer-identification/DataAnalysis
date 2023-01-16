# DataAnalysis
Repository for analysis of data

## Data
Data is organized into 6 versions, every version has three stages
### Raw
Raw files, obtained from firebase in json format
### interim
Files, converted to csv format

### processed
Feature sets with calculcated features, ready for data analysis and machine learning. Version generally corespond to version in previous stages, with the exception of V6, which is a combination of V4, V5 and V6 datasets. V6 feature sets were used for data analysis and training in paper.

Every feature set is in two files:
- .csv file with a actual cases and their features
- .json file with feature set metadata

Since some of the final feature sets are huge (>100MB in size) and are over github's limit, they are missing in directory (only .json files present). csv files are availabel on this [link](https://drive.google.com/drive/folders/1fYVc1yhNmNVduCusNwB--1vaS4BON5VF?usp=sharing)

## Jupyter notebooks
notebooks are organised into versions according to when during the project they were used. Most of the code for the final paper is in 5.0 playbooks with some exceptions, which are still part of 4.0 (generate-full-fourier-statistical-dataset.ipynb, generate_fourier_dataset.ipynb)
