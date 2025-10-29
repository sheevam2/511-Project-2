# Part 2: Benchmarking HNSW with Increasing Dataset Sizes

## Description:
This part evaluates the scalability of HNSW as the dataset size increases by analyzing QPS, Recall, and Index Build Time across datasets of varying sizes.

## Requirements:
- Download 4 different datasets from ann-benchmarks
- Vary M parameter across [4, 8, 12, 24, 48]
- Tune efSearch and efConstruction parameters for each dataset
- Generate QPS vs Recall and Index Build Time vs Recall plots

## Datasets to Use:
Choose 4 datasets with increasing sizes from:
- MNIST (60K vectors, 784 dim)
- NYTimes (290K vectors, 256 dim)
- GloVe-25 (1.18M vectors, 25 dim)
- GloVe-50 (1.18M vectors, 50 dim)
- GloVe-100 (1.18M vectors, 100 dim)
- SIFT (1M vectors, 128 dim)

## Deliverables:
- QPS vs Recall plot (different curves for different dataset sizes)
- Index Build Time vs Recall plot (different curves for different dataset sizes)
- Analysis of parameter impacts and dataset properties

## Status:
🚧 **Not implemented yet** - Ready for development
