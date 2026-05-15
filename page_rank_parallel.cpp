
#include <cstdio>
#include "core/utils.h"
#include "core/graph.h"
#include <mpi.h>
#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
#define PAGERANK_MPI_TYPE MPI_LONG
#define PR_FMT "%ld"
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
#define PAGERANK_MPI_TYPE MPI_FLOAT
#define PR_FMT "%f"
typedef float PageRankType;
#endif

struct Results
{
  uintE num_edges;
  double total_time;
  int rank;
};

void pageRankSerial(Graph &g, int max_iters)
{
  uintV n = g.n_;

  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];

  for (uintV i = 0; i < n; i++)
  {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  timer t1;
  double time_taken = 0.0;
  t1.start();
  for (int iter = 0; iter < max_iters; iter++)
  {
    for (uintV u = 0; u < n; u++)
    {
      uintE out_degree = g.vertices_[u].getOutDegree();
      for (uintE i = 0; i < out_degree; i++)
      {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        pr_next[v] += (pr_curr[u] / out_degree);
      }
    }
    for (uintV v = 0; v < n; v++)
    {
      pr_next[v] = PAGE_RANK(pr_next[v]);
      pr_curr[v] = pr_next[v];
      pr_next[v] = 0.0;
    }
  }
  time_taken = t1.stop();

  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++)
  {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
  std::cout << "Time taken (in seconds) : " << time_taken << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

// ============================= Strat 1 ===============================
void EdgeDecompostion(std::vector<uintV> &startV, std::vector<uintV> &endV, Graph &g, uintV n, int world_size, int world_rank)
{

  uintV start_vertex = 0;
  uintV end_vertex = 0;
  uintE m = g.m_;

  for (int i = 0; i < world_size; i++)
  {
    start_vertex = end_vertex;
    long count = 0;
    while (end_vertex < n)
    {

      count += g.vertices_[end_vertex].getOutDegree();
      end_vertex += 1;
      if (count >= m / world_size)
        break;
    }
    startV[i] = start_vertex;
    endV[i] = end_vertex;
  }
}

void ReduceAndScatter(int world_size, int world_rank, int n, PageRankType *p_input, std::vector<uintV> &startV, std::vector<uintV> &range,std::vector<PageRankType> &p_reduce)
{


  if (world_rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, p_input, n, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
    std::copy(p_input, p_input + n, p_reduce.begin());
  }
  else
  {
    MPI_Reduce(p_input, nullptr, n, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  MPI_Scatterv(world_rank == 0 ? p_reduce.data() : nullptr, range.data(), startV.data(), PAGERANK_MPI_TYPE, p_input + startV[world_rank], range[world_rank], PAGERANK_MPI_TYPE, 0, MPI_COMM_WORLD);
}



Results PageRank1(PageRankType *pr_curr, PageRankType *pr_next, uintV n, int max_iters, Graph &g, std::vector<uintV> &startV, std::vector<uintV> &endV, int world_rank, int world_size)
{

  timer t1;
  uintE num_edges = 0;
  double total_time = 0.0;
  uintV start = startV[world_rank];
  uintV end = endV[world_rank];
  std::vector<uintV> range(world_size);
  std::vector<PageRankType> p_reduce(n);
  for (int i = 0; i < world_size; i++)
  {
    range[i] = endV[i] - startV[i];
  }
 
  for (int iter = 0; iter < max_iters; iter++)
  {
    // for each vertex 'u', process all its outNeighbors 'v'
    for (uintV u = start; u < end; u++)
    {

      uintE out_degree = g.vertices_[u].getOutDegree();
      if (out_degree == 0)
      {
        continue;
      }
      PageRankType temp = (pr_curr[u] / out_degree);
      for (uintE i = 0; i < out_degree; i++)
      {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        pr_next[v] += temp;
        num_edges++;
      }
    }

    // compute time for the syncing
    t1.start();
   
    ReduceAndScatter(world_size, world_rank, n, pr_next, startV, range,p_reduce);

    total_time += t1.stop();

    // reset and get PAGE_RANK
    
    for (uintV v = 0; v < n; v++)
    {
      PageRankType val = pr_next[v];
      pr_curr[v] = PAGE_RANK(val);
      pr_next[v] = 0.0;
    }
    
  }

  Results result = {num_edges, total_time, world_rank};
  return result;
}


void MPI_Call(const int world_rank, const int world_size, Graph &g, int max_iters)
{
  // set up
  uintV n = g.n_;

  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];
  for (uintV i = 0; i < n; i++)
  {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }
  std::vector<Results> results(world_size);
  PageRankType sumPages = 0.0;
  // time setup
  timer t1;
  double time_taken = 0.0;
  t1.start();

  // edge decomp still have a little to do in function

  std::vector<uintV> start(world_size);
  std::vector<uintV> end(world_size);
  EdgeDecompostion(start, end, g, n, world_size, world_rank);

  // call PageRank
  Results local_result = PageRank1(pr_curr, pr_next, n, max_iters, g, start, end, world_rank, world_size);

  time_taken = t1.stop();
  std::printf("%d, %d, %f\n",world_rank,local_result.num_edges,local_result.total_time);

  if (world_rank == 0)
  {
    for (uintV u = 0; u < n; u++)
    {
      sumPages += pr_curr[u];
    }
  

    std::printf("Sum of page rank : " PR_FMT "\n", sumPages);

    printf("Time taken (in seconds) :  %f\n",time_taken);
  }

  delete[] pr_curr;
  delete[] pr_next;
}
// ======================== Strat 2 ================================================================================================================================================================================
//==============================================================================================================================================================================================
//==============================================================================================================================================================================================
void ReduceToAll(int world_size, int world_rank, int n, PageRankType *p_input, std::vector<uintV> &startV, std::vector<uintV> &range)
{

  
  for (int i = 0; i < world_size; i++)
  {
    if (world_rank == i)
    {
      std::vector<PageRankType> p_reduce;
      MPI_Reduce(MPI_IN_PLACE, p_input + startV[i], range[i], PAGERANK_MPI_TYPE, MPI_SUM, i, MPI_COMM_WORLD);
    }
    else
    {
      MPI_Reduce(p_input + startV[i], nullptr, range[i], PAGERANK_MPI_TYPE, MPI_SUM, i, MPI_COMM_WORLD);
    }
  }
}

Results PageRank2(PageRankType *pr_curr, PageRankType *pr_next, uintV n, int max_iters, Graph &g, std::vector<uintV> &startV, std::vector<uintV> &endV, int world_rank, int world_size)
{

  timer t1;
  uintE num_edges = 0;
  double total_time = 0.0;
  uintV start = startV[world_rank];
  uintV end = endV[world_rank];
  std::vector<uintV> range(world_size);
  for (int i = 0; i < world_size; i++)
  {
    range[i] = endV[i] - startV[i];
  }

  for (int iter = 0; iter < max_iters; iter++)
  {
    // for each vertex 'u', process all its outNeighbors 'v'
    for (uintV u = start; u < end; u++)
    {

      uintE out_degree = g.vertices_[u].getOutDegree();
      if (out_degree == 0)
      {
        continue;
      }
      PageRankType temp = (pr_curr[u] / out_degree);
      for (uintE i = 0; i < out_degree; i++)
      {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        pr_next[v] += temp;
        num_edges++;
      }
    }

    // compute time for the syncing
    t1.start();
    
    ReduceToAll(world_size, world_rank, n, pr_next, startV, range);

    total_time += t1.stop();

    
     for (uintV v = start; v < end; v++)
      {
        PageRankType val = pr_next[v];
        pr_curr[v] = PAGE_RANK(val);
      }
      for (uintV v = 0; v < n; v++)
      {
        pr_next[v] = 0.0;
      }
    
  }

  Results result = {num_edges, total_time, world_rank};
  return result;
}


void MPI_Call_S2(const int world_rank, const int world_size, Graph &g, int max_iters){
  // set up
  uintV n = g.n_;

  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];
  for (uintV i = 0; i < n; i++)
  {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }
  std::vector<Results> results(world_size);
  PageRankType sumPages = 0.0;
  // time setup
  timer t1;
  double time_taken = 0.0;
  t1.start();

  // edge decomp still have a little to do in function

  std::vector<uintV> start(world_size);
  std::vector<uintV> end(world_size);
  EdgeDecompostion(start, end, g, n, world_size, world_rank);


  Results local_result = PageRank2(pr_curr, pr_next, n, max_iters, g, start, end, world_rank, world_size);
  PageRankType result = 0.0;

  PageRankType sum = 0.0;
  for (int i = start[world_rank]; i < end[world_rank]; i++)
  {
    sum += pr_curr[i];
  }
  MPI_Reduce(&sum, &result, 1, PAGERANK_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
  time_taken = t1.stop();
  std::printf("%d, %d, %f\n",world_rank,local_result.num_edges,local_result.total_time);
  


  if (world_rank == 0)
  {

    std::printf("Sum of page rank : " PR_FMT "\n", result);

    printf("Time taken (in seconds) :  %f\n",time_taken);
  }

  delete[] pr_curr;
  delete[] pr_next;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  cxxopts::Options options("page_rank_push", "Calculate page_rank using serial and parallel execution");
  options.add_options("", {
                              {"nIterations", "Maximum number of iterations", cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
                              {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                              {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
                          });

  auto cl_options = options.parse(argc, argv);
  uint strategy = cl_options["strategy"].as<uint>();
  uint max_iterations = cl_options["nIterations"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();

#ifdef USE_INT
  std::printf("Using INT\n");
#else
  std::printf("Using FLOAT\n");
#endif
  // Get the world size and print it out here
  if (world_rank == 0)
  {
    std::printf("World size : %d\n", world_size);
    std::printf("Communication strategy : %d\n", strategy);
    std::printf("Iterations : %d\n", max_iterations);
  }

  Graph g;
  g.readGraphFromBinary<int>(input_file_path);
  switch (strategy)
  {
  case 0:
    pageRankSerial(g, max_iterations);
    break;
  case 1:
    MPI_Call(world_rank, world_size, g, max_iterations);
    break;
  case 2:
    MPI_Call_S2(world_rank, world_size, g, max_iterations);
    break;
  default:
    break;
  }

  // pageRankSerial(g, max_iterations);
  MPI_Finalize();
  return 0;
}

