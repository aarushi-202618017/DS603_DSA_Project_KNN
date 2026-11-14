# **DS603 — Data Structures and Algorithms | Group 9**

# Nearest-Neighbor Search Engine using KD-Tree

A Python Streamlit application that compares a custom KD-Tree with brute-force K-Nearest Neighbors (KNN) for forest cover classification.

**Dashboard Link: https://ds603-dsaproject-grp09-knn-kdtree-bruteforce.streamlit.app/**

## Features

- KD-Tree implemented from scratch using median splitting, recursive search, pruning, and a max-heap.
- Brute-force KNN using NumPy.
- Interactive forest cover classifier and nearest-neighbor recommendations.
- Performance comparison across dataset sizes and dimensions.

## Dataset

**Forest CoverType — UCI Machine Learning Repository**

- 120,000 records in the included CSV
- 10 continuous terrain features
- 7 forest cover classes
- 80/20 stratified train-test split
- StandardScaler for feature normalization

## Technologies

Python, Streamlit, NumPy, pandas, scikit-learn, matplotlib, ucimlrepo.

## Installation

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```


## Benchmark Results

| Dimensions | Brute Force | KD-Tree | Improvement |
|------------|-------------|---------|-------------|
| 2D | 471.76 ms | 27.13 ms | 94.2% |
| 5D | 830.92 ms | 214.23 ms | 74.2% |
| 10D | 1115.04 ms | 826.88 ms | 25.8% |

Results represent average query batch times across five training sizes, with 100 queries per experiment.

## Team — Group 9

| Members | Contribution |
|---------|--------------|
| Twinkle, Palak | Dataset preprocessing, cleaning, feature selection |
| Shreya, Hemangi | KD-Tree and KD-Tree KNN implementation |
| Yatri | Brute-force KNN implementation |
| Aarushi, Gauri | Performance comparison, Streamlit UI, documentation |

## Dataset Reference

[UCI Forest CoverType](https://archive.ics.uci.edu/dataset/31/covertype)
