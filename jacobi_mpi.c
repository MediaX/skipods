/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <mpi.h>

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n,
   float A[ n],
   float B[ n])
{
  int i;

  for (i = 0; i < n; i++)
      {
 A[i] = ((float) i+ 2) / n;
 B[i] = ((float) i+ 3) / n;
      }
}

static
void print_array(int n,
   float A[ n])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", A[i]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_jacobi_1d(int tsteps,
       int n,
       float A[ n],
       float B[ n],
       int size,
       int rank)
{
  int t, i;
  int num, ibeg, iend;
  num = ((n - 2) - 1) / size + 1;
  ibeg = rank * num;
	iend = ibeg + num;
  if (iend > n){
    iend = n;
  }
  for (t = 0; t < tsteps; t++)
    {
      float C[n];
      if (rank = 0){
        for (i = 0; i < n; i++)
          C[i] = 0;
      }

      MPI_Bcast(A, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(B, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

      for (i = ibeg + 1; i < iend - 1; i++)
 B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);

      MPI_Reduce(B, C, n, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
      memcpy(B, C, n*sizeof(float));

      MPI_Bcast(A, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(B, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

      for (i = ibeg + 1; i < iend - 1; i++)
 A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
      MPI_Reduce(A, C, n, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
      memcpy(A, C, n*sizeof(float));
    }
}


int main(int argc, char** argv)
{
  int n = N;
  int tsteps = TSTEPS;
  int rank, size;
  int count = 0;
  double time_start = 0, time_end = 0;
#ifdef MINI_DATASET
  FILE *f = fopen("result_mini.txt", "a+");
#endif
#ifdef SMALL_DATASET
  FILE *f = fopen("result_small.txt", "a+");
#endif
#ifdef MEDIUM_DATASET
  FILE *f = fopen("result_medium.txt", "a+");
#endif
#ifdef LARGE_DATASET
  FILE *f = fopen("result_large.txt", "a+");
#endif
#ifdef EXTRALARGE_DATASET
  FILE *f = fopen("result_extralarge.txt", "a+");
#endif

  float (*A)[n]; A = (float(*)[n])malloc ((n) * sizeof(float));;
  float (*B)[n]; B = (float(*)[n])malloc ((n) * sizeof(float));;

  init_array (n, *A, *B);

  // bench_timer_start();;

  MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);
  time_start = MPI_Wtime();

  kernel_jacobi_1d(tsteps, n, *A, *B, size, rank);

  MPI_Barrier(MPI_COMM_WORLD);
  time_end = MPI_Wtime();

  // bench_timer_stop();;
  // bench_timer_print();;

  //if (argc > 42 && ! strcmp(argv[0], "")) print_array(n, *A);
  if (rank == 0){
    print_array(n, *A);
    fprintf(f, "\nnum of proc %d", size);
    fprintf(f, "\nTime of work %0.6lf\n", time_end - time_start);
  }
  MPI_Finalize();
  free((void*)A);;
  free((void*)B);;

  return 0;
}
