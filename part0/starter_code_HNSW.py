import faiss
import h5py
import numpy as np
import os
import requests
import time

def evaluate_hnsw():
    """
    Part 0: Create HNSW index and perform queries on SIFT1M dataset
    """
    
    # Load the SIFT1M dataset
    print("Loading SIFT1M dataset...")
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
        train_embeddings = f['train'][:]  # Database vectors
        test_embeddings = f['test'][:]    # Query vectors
    
    print(f"Loaded {len(train_embeddings)} training vectors and {len(test_embeddings)} test vectors")
    print(f"Vector dimension: {train_embeddings.shape[1]}")
    
    # Create HNSW index with specified parameters
    print("Creating HNSW index...")
    dimension = train_embeddings.shape[1]
    
    # Create HNSW index with M=16, efConstruction=200
    index = faiss.IndexHNSWFlat(dimension, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200
    
    # Add training vectors to the index
    print("Adding vectors to index...")
    start_time = time.time()
    index.add(train_embeddings.astype('float32'))
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds")
    
    # Perform query using the first test vector
    print("Performing query...")
    query_vector = test_embeddings[0:1].astype('float32')  # First query vector
    
    # Search for top 10 nearest neighbors
    start_time = time.time()
    distances, indices = index.search(query_vector, 10)
    query_time = time.time() - start_time
    
    print(f"Query completed in {query_time:.4f} seconds")
    print(f"Top 10 nearest neighbor indices: {indices[0]}")
    
    # Write results to output.txt
    print("Writing results to output.txt...")
    with open('output.txt', 'w') as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")
    
    print("Results written to output.txt")
    print(f"Index contains {index.ntotal} vectors")
    print(f"Query returned distances: {distances[0]}")

if __name__ == "__main__":
    evaluate_hnsw()
