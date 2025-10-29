# Part 1: HNSW vs LSH Comparison

## Files:
- `part1_hnsw_vs_lsh.py` - Main implementation file
- `part1_qps_vs_recall_matplotlib.png` - Generated plot comparing HNSW vs LSH
- `part1_results.json` - Detailed results data

## Description:
This part compares the performance of HNSW and LSH algorithms on the SIFT1M dataset by measuring 1-Recall@1 and QPS under different parameter configurations.

## Parameters Tested:

### HNSW:
- M = 32, efConstruction = 200
- efSearch values: [10, 50, 100, 200]

### LSH:
- nbits values: [32, 64, 512, 768]

## Results Summary:

### HNSW Performance:
- efSearch=10: QPS=154,089, Recall=84.1%
- efSearch=50: QPS=51,794, Recall=98.1%
- efSearch=100: QPS=29,160, Recall=99.3%
- efSearch=200: QPS=15,926, Recall=99.5%

### LSH Performance:
- nbits=32: QPS=8,782, Recall=1.5%
- nbits=64: QPS=8,269, Recall=4.9%
- nbits=512: QPS=4,107, Recall=31.5%
- nbits=768: QPS=1,771, Recall=37.3%

## How to Run:
```bash
cd part1
python part1_hnsw_vs_lsh.py
```

Note: Requires the SIFT1M dataset (`sift-128-euclidean.hdf5`) to be in the parent directory.
