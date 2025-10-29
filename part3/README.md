# Part 3: HNSW vs DiskANN Comparison

## Description:
This part compares the performance of DiskANN and HNSW by plotting Latency vs Recall on SIFT1M dataset, evaluating how varying key parameters impacts their performance.

## Requirements:
- Use actual DiskANN library (not FAISS simulation)
- Compare on SIFT1M dataset
- Plot Latency vs Recall for both algorithms
- Vary parameters for both algorithms

## DiskANN Parameters to Vary:
- Graph Degree (R): Controls neighbors per node
- Complexity (L): Size of search list
- In-Memory Size: RAM allocation
- Disk Memory Size: Disk space allocation

## HNSW Parameters to Vary:
- M, efConstruction, efSearch

## Deliverables:
- Latency vs Recall plot for both algorithms
- Analysis of parameter effects
- Comparison of when to use each algorithm

## Status:
🚧 **Not implemented yet** - Ready for development

## Note:
Requires installing the actual DiskANN library from Microsoft's repository.
