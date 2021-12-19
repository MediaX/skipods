/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <omp.h>

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
       int num_thr)
{
  int t, i;
#pragma omp parallel for private(t, i) shared(A, B) num_threads(num_thr)
  for (t = 0; t < tsteps; t++)
    {
      for (i = 1; i < n - 1; i++)
 B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < n - 1; i++)
 A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
}


int main(int argc, char** argv)
{
  int n = N, i = 0;
  int tsteps = TSTEPS;
  double times[3];
  // sscanf(argv[3], "%d", &n);
  float (*A)[n]; A = (float(*)[n])malloc ((n) * sizeof(float));;
  float (*B)[n]; B = (float(*)[n])malloc ((n) * sizeof(float));;
  int num_thr;
  int j = 0;
  double time;
  sscanf(argv[1], "%d", &num_thr);
  // sscanf(argv[2], "%d", &tsteps);
  for (j = 1; j <= 128; j*=2){
    printf("for %d threads\n", j);
    for (i = 0; i < 3; i++){
      init_array (n, *A, *B);

      bench_timer_start();
      omp_set_num_threads(j);
      time = omp_get_wtime();	
      kernel_jacobi_1d(tsteps, n, *A, *B, j);
      times[i] = omp_get_wtime() - time;
      bench_timer_stop();
      //bench_timer_print();
    }
    i = 0;
    printf ("Average time in seconds = %0.6lf\n", (times[0]+times[1]+times[2])/3);
    printf("\n\n\n");
  }

  // fprintf(stderr, "%d\n", strcmp(argv[0], ""));
  // if (argc > 42 && ! strcmp(argv[0], "")) print_array(n, *A);
  //if (0)
  //  print_array(n, *A);


  free((void*)A);;
  free((void*)B);;

  return 0;
}
