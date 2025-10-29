# Part 0: Creating Index and Performing Queries

## Files:
- `starter_code_HNSW.py` - Main implementation file
- `output.txt` - Results file with top 10 nearest neighbor indices

## Description:
This part creates an HNSW index on the SIFT1M dataset and performs a query to find the top 10 nearest neighbors.

## Parameters Used:
- M = 16
- efConstruction = 200
- efSearch = 200

## Results:
Top 10 nearest neighbor indices: [932085, 934876, 561813, 708177, 706771, 695756, 435345, 701258, 455537, 872728]

## How to Run:
```bash
cd part0
python starter_code_HNSW.py
```

Note: Requires the SIFT1M dataset (`sift-128-euclidean.hdf5`) to be in the parent directory.
