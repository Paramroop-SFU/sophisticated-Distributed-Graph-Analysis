# sophisticated-Distributed-Graph-Analysis
# Distributed Graph Analysis Suite 🌐

High-performance distributed PageRank and Triangle Counting using C++ and MPI.

## 🚀 Key Strategies
* **Edge Partitioning:** Ensures O(m/P) workload distribution.
* **Communication Optimization:** Toggle between `MPI_Gather`, `MPI_Reduce`, and `MPI_Scatterv` based on strategy flags.
* **Numerical Precision:** Configurable `USE_INT` for high-throughput integer math.

## 🛠 Usage
```bash
# Compile
make

# Run Triangle Counting with 4 nodes
mpirun -np 4 ./triangle_counting_mpi --strategy 1 --inputFile roadNet-CA

# Run PageRank with 8 nodes
mpirun -np 8 ./page_rank_mpi --strategy 2 --nIterations 20
