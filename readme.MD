# Research Paper: Exploring Feature Selection with Deep Learning for Kidney Tissue Microarray Classification using Infrared Spectral Imaging

This repository contains the code and resources for the research paper titled "Exploring Feature Selection with Deep Learning for Kidney Tissue Microarray Classification using Infrared Spectral Imaging". The paper focuses on the classification of two different types of renal cancer: Chromophobe Renal Cell Carcinoma (ChrCC) and Oncocytoma.

## Abstract

Kidney and renal pelvic cancer are a significant cause of cancer-related deaths, with the most common malignant kidney tumor being renal cell carcinoma (RCC). Chromophobe renal cell carcinoma is a rarer form of RCC, that poses significant challenges to accurate diagnosis, as it shares many histologic features with Oncocytoma, a benign renal tumor. Biopsies for histopathological and immunohistochemical analysis have limitations in distinguishing Chromophobe RCC and Oncocytoma. Syndromic cases may also have tumors with overlapping features. Techniques such as Infrared (IR) spectroscopic imaging have shown promise as an alternate approach to tissue diagnostics.

In this study, we propose a deep learning-based framework for automating classification in kidney tumor tissue microarrays (TMAs), using an IR dataset. Feature selection algorithms reduce data dimensionality followed by a deep learning classification approach. Classification accuracy of 95.47% was observed for validation data even with using 13.56% of all wavelengths thereby reducing training time by 21% compared to using the entire spectrum. Through the integration of scalable deep learning models coupled with feature selection, we have developed a classification pipeline with high predictive power which could be integrated into a high throughput real-time IR imaging system. This would create an advanced diagnostic tool for detection and classification of renal tumors namely Chromophobe RCC and Oncocytoma. This may impact patient outcome and treatment strategies.

## Repository Structure

- `data/`: We are unable to share the data due to the UIC IRB #2016-0581. 
- `FS/`: Contains the code for feature selection.
- `NN/`: Contains the code and results from the Neural Networks.

## Citation

If you find this research paper or the code in this repository useful, please consider citing:

[Insert citation here]

## Funding

This document is the results of the research project funded by the UW-Eau Claire Office of Research and Sponsored Programs. The computational resources of the study were provided by the Blugold Center for High-Performance Computing under National Science Foundation grant CNS-1920220.

## All Thanks To Our Contributors:

<a href="https://github.com/caterer-z-t/AI-Bioimaging/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=caterer-z-t/AI-Bioimaging" />
</a>
