# SSRF

[![PyPI version](https://img.shields.io/pypi/v/ssrf.svg)](https://pypi.org/project/ssrf/)
[![Python Version](https://img.shields.io/pypi/pyversions/ssrf.svg)](https://pypi.org/project/ssrf/)
[![License](https://img.shields.io/pypi/l/ssrf.svg)](LICENSE)

SSRF (Spatial-Spectral Random Forest) is a simple but effective method for gap-filling thick 
cloud pixels in satellite (e.g., Landsat, Sentinel-2) image pixels. The package implments SSRF
based on Wang et al. (2022) but with XGBoost, numba and dask for improved speed and scalability.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Example](#quick-example)
- [Usage](#usage)
- [License](#license)
- [Citation](#citation)

---

## Introduction

SSRF (Spatial-Spectral Random Forest) is a simple but effective method for gap-filling thick 
cloud pixels in satellite (e.g., Landsat, Sentinel-2) image pixels (Wang et al., 2022). The 
technique uses spatial spatially adjacent and multispectral information of known images 
simultaneously based on random forests. However, the technique is notoriously memory-heavy and 
slow in Python. This implementation is virtually the same as the original implementation 
(Wang et al. 2022) except we use XGBoost, numba, and dask (optional) to reduce processing time 
improve scalability. 

Key features:
- Modern gap-filling algorithm;
- Works on multi-spectral reflectance or single-band indices (e.g., NDVI);
- Fast and scalable implemntation based on numba and dask;
- Easy to integrate with existing Python workflows
- Minimal dependencies;

---

## Installation


## Citation
If you use SSRF in a publication, please cite original authors: 
Wang, Q., Wang, L., Zhu, X., Ge, Y., Tong, X., & Atkinson, P.M. (2022). *Remote sensing image gap filling based on spatial-spectral random forests*. Science of Remote Sensing, 5, 100048. https://doi.org/10.1016/j.srs.2022.100048