#include <iostream>
#include <cstdio>
#include "core/utils.h"
#include "core/graph.h"
#include <mpi.h>

using namespace std;
struct Result
{
  long triangle;
  double time;
  uintE num_edges;
  int rank;
};



long countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2,
                    uintV u, uintV v)
{
  uintE i = 0, j = 0; // indexes for array1 and array2
  long count = 0;

  if (u == v)
    return count;

  while ((i < len1) && (j < len2))
  {
    if (array1[i] == array2[j])
    {
      if ((array1[i] != u) && (array1[i] != v))
      {
        count++;
      }
      else
      {
        // triangle with self-referential edge -> ignore
      }
      i++;
      j++;
    }
    else if (array1[i] < array2[j])
    {
      i++;
    }
    else
    {
      j++;
    }
  }
  return count;
}
void triangleCountSerial(Graph &g) {
  uintV n = g.n_;
  long triangle_count = 0;
  double time_taken = 0.0;
  timer t1;
  t1.start();
  for (uintV u = 0; u < n; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
    }
  }
  time_taken = t1.stop();
  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << time_taken << "\n";
}


void EdgeDecompostion(uintV&startV,uintV&endV,Graph&g,uintV n,int world_size,int world_rank){

  uintV start_vertex = 0;
  uintV end_vertex = 0;
  uintE m = g.m_;

  for (int i = 0; i < world_size; i++) {
      start_vertex = end_vertex;
      long count = 0;
      while (end_vertex < n)
      {
       
          count += g.vertices_[end_vertex].getOutDegree();
          end_vertex += 1;
          if (count >= m / world_size)
              break;
      }
      if(i == world_rank)
        break;
      
  }
  startV = start_vertex;
    endV   = end_vertex;
}


Result triangleCountLocal(Graph &g, uintV start, uintV end, vector<uintE> &edges, uintV n, int rank)
{
  timer t1;
  t1.start();

  // thread_id, num_vertices, num_edges, triangle_count, time_taken
  long triangle_count = 0;
  uintE num_edges = 0;
  uintV num_vertices = 0;
  uintV temp_start = start;

  for (uintV u = start; u < end; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
      num_edges++;
    }
  }
  double time_taken = t1.stop();
  Result r = {triangle_count, time_taken, num_edges, rank};
  return r;
}

long SyncWithGather(int world_rank,int world_size,Result& local_result){
  long triangle_count = 0;
  Result * results;
  if (world_rank == 0){
    results = new Result[world_size];
  }
  MPI_Gather(&local_result,sizeof(Result),MPI_BYTE,results,sizeof(Result),MPI_BYTE,0,MPI_COMM_WORLD);
  if (world_rank == 0)
  {

    printf("rank, edges, triangle_count, communication_time\n");
    for (int i = 0; i < world_size; i++)
    {  

      printf("%i, %i, %ld, %lf\n",results[i].rank,results[i].num_edges,results[i].triangle,results[i].time);
      triangle_count += results[i].triangle;
    }

    delete[] results;
  }
   
   return triangle_count;
}

long syncwithReduce(int world_rank,int world_size,Result& local_result){
  long sum = 0;
  if (world_rank ==0){
    printf("rank, edges, triangle_count, communication_time\n");
  }
  printf("%i, %i, %ld, %lf\n",world_rank,local_result.num_edges,local_result.triangle,local_result.time);
  MPI_Reduce(&local_result.triangle,&sum,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
  return (world_rank == 0) ? sum : 0;

}

void MPI_Call(const int world_rank, int world_size, Graph &g,long (*func)(int world_rank,int world_size,Result& local_result))
{
  // do the local work first then send info

  uintV n = g.n_;
  long triangle_count = 0;
  double time_taken = 0.0;
  timer t1;
  t1.start();
  vector<uintE> edges(n + 1, 0);
  uintE edge_count = 0;
  
  uintV nums_per_thread = edge_count / world_size;
  uintV start;
  uintV end; 
  EdgeDecompostion(start,end,g,n,world_size,world_rank);
  // i have the start and end
  Result local_result = triangleCountLocal(g, start, end, edges, n, world_rank);

  triangle_count =  func(world_rank,world_size,local_result);
  

  time_taken = t1.stop();

  // Print out overall statistics
  if (world_rank == 0)
  {
    std::printf("Number of triangles : %ld\n", triangle_count);
    std::printf("Number of unique triangles : %ld\n", triangle_count / 3);
    std::printf("Time taken (in seconds) : %f\n", time_taken);
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options("triangle_counting_serial", "Count the number of triangles using serial and parallel execution");
  options.add_options("custom", {
                                    {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                    {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
                                });

  auto cl_options = options.parse(argc, argv);
  uint strategy = cl_options["strategy"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();
  Graph g;
  g.readGraphFromBinary<int>(input_file_path);
  if (world_rank == 0 && strategy >= 1)
  {
    std::printf("World size : %d\n", world_size);
    std::printf("Communication strategy : %d\n", strategy);
  }

  // Get the world size and print it out here
  switch (strategy)
  {
  case 0:
    triangleCountSerial(g);
    break;
  case 1:
    MPI_Call(world_rank, world_size, g,SyncWithGather);
    break;
  case 2:
    MPI_Call(world_rank, world_size, g,syncwithReduce);
    break;

  default:
    break;
  }

 


  
  // triangleCountSerial(g);
  MPI_Finalize();
  return 0;
}
