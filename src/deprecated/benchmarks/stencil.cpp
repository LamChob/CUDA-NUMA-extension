/*
 * Exercise 2 - Heat Distribution
 * main.cpp
 *  Created on: 29.10.2015
 *      Author: Dennis Rieber and Jan Ringkamp
 *      Group: pra07
 *      Purpose: Calculate the heat distribution on a NxN square over several iteration steps.
 */

//#include <Windows.h> use only on windows systems
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <NUMA_Container.h>

#include <pthread.h>

int N_IT;

typedef std::vector<double> Dim1;
typedef std::vector<Dim1> Mat_2D;

typedef struct t_dat {
    int t_id, no_t, size, no_it;
    NUMA_Container<float>* mat_old;
    NUMA_Container<float>* mat_new;
} thread_data;

pthread_barrier_t barr;
std::string res_dir, mkdir_string, cmd_line;


void print_mat( int h, int w, Mat_2D mat) 
{
        std::cout << std::endl;
    for ( int i = 0; i<h; ++i)
    {
        for ( int j = 0; j<w;++j)
        {
            std::cout << mat[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}
void pin_thread( int core ) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET( core , &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void calc_workspace( t_dat* t )
{
    pin_thread(t->t_id * 6);
    int w_size = t->size / t->no_t;
    int index_start = t->t_id * w_size;
    int index_end   = ((t->t_id+1) * w_size ) - 1;
    if ( t->t_id == 0 ) ++index_start;
    if ( t->t_id == t->no_t-1) --index_end;
    double avg_wait = 0.0;

    
    struct timeval start, end;
    for ( int k = 0; k < N_IT; ++k )
    {
        for ( int i = index_start; i <= index_end; ++i ) 
        {
            int row = i * t->size;
            for ( int j = 1; j < (t->size - 1); ++j )
            {
                t->mat_new->operator[](row + j) =  t->mat_old->operator[](row + j) + 6.0/25.0 * ((-4.0) * (
                    t->mat_old->operator[](row + j) + 
                    t->mat_old->operator[](row + j + 1) + 
                    t->mat_old->operator[](row + t->size + j) + 
                    t->mat_old->operator[](row + j-1) +
                    t->mat_old->operator[](row - t->size + j)
                ));
            } 
        }
        // synchronize all threads before swapping pointers 
        
        gettimeofday(&start, 0);
        int rc = pthread_barrier_wait(&barr);
        gettimeofday(&end, 0);
        avg_wait += end.tv_usec - start.tv_usec;

        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(-1);
        }

        std::swap(t->mat_new, t->mat_old);
    } 
}

// function that inits multiplication
void *thr_func( void *arg ) 
{
    // calculate worksize
    t_dat* args = (thread_data*) arg;
    calc_workspace(args);
}

int main(int argc, char* argv[])
{
	int N, m, N_TR;
	double H;
	std::ostringstream convert;

	if(argc != 6)
	{
		std::cout << "Wrong number of arguments" << std::endl;
		std::cout << "Please enter the following arguments:" << std::endl;
		std::cout << "N  heat distribution will be computed across a square of NxN element" << std::endl;
		std::cout << "m  radius of initial homogeneous circular heat source (max: N/2 - 1)" << std::endl;
		std::cout << "H  temperature of initial heat source in degree Celsius (decimal place possible, max: 127.0)" << std::endl;
		std::cout << "NT Number of threads to start " << std::endl;
		return -1;
	}

	N = atoi(argv[1]);
	m = atoi(argv[2]);
	H = atoi(argv[3]);
        N_TR = atoi(argv[4]);
        N_IT = atoi(argv[5]);

	// check if the data range is valid
	bool err_flag = false;

	if((m < 1) || (m > (N/2 - 1)))
	{
	    err_flag = true;
		std::cout << "m is outside of the valid region: 1 to (N/2-1)" << std::endl;
	}

	if((H < 0.0) || (H > 127.0))
    {
        err_flag = true;
        std::cout << "H is outside of the valid region: 0.0 to 127.0" << std::endl;
    }

	if(err_flag)
	{
	    return -1;
	}


	//output parameters
	/*std::cout << "Calculating heat distribution for a " << N << "x" << N << " square" << std::endl;
	std::cout << "Radius of initial heat source: " << m << std::endl;
	std::cout << "Temperature of initial heat source: " << H << std::endl;*/

	//generate two NxN matrices, initialized with zeros

    NUMA_Container<float> mat_old(N*N);
    NUMA_Container<float> mat_new(N*N);

	//calculate center of circle
	double cx, cy, dist;
	cx = N/2;
	cy = N/2;

	//initiate matrix
	for(int i = 0; i < N; i++)
	{
        int row = i * N;
		for(int j = 0; j < N; j++)
		{
			//calculate the distance of the current elements center to the matrix center
			dist = sqrt(pow((cx - (i + 0.5)),2) + pow((cy - (j + 0.5)),2));
			//check if center of current element is inside the circle
			if(dist <= m)
			{
				mat_old[row + j] = H;
			} else {
                mat_old[row + j] = 0;
            }
            mat_new[row + j] = 0;
		}

	}


	//compute heat distribution until N_IT iterations are reached
	struct timeval start, end;
	double avg_cp_time = 0;
    // Thread args and thread container
    t_dat* args    = new t_dat[N_TR]; 
    pthread_t* thr = new pthread_t[N_TR]; 

    if ( pthread_barrier_init(&barr, NULL, N_TR))
    {
        std::cout << "Failed to create Barrier" << std::endl;
        exit(-1);
    }
	if ( N_TR == 1 ){}
	else if ( N_TR == 2 ) {
	    mat_old.distribute({0, 1});
            mat_new.distribute({0, 1});
	}
	else if ( N_TR == 4 ) {
	    mat_old.distribute({0, 1, 2, 3});
            mat_new.distribute({0, 1, 2, 3});
	}
	else {
	    mat_old.distribute({0, 1, 2, 3, 4, 5, 6 ,7});
            mat_new.distribute({0, 1, 2, 3, 4, 6, 6, 7});
	}
	gettimeofday(&start,0);
	
    /* fork here */
    for ( int i = 0; i < N_TR; ++i ) 
    {
        // parameters
        args[i].size = N;
        args[i].no_t = N_TR;
        args[i].t_id = i;
        args[i].no_it = N_IT; 

        // data
        args[i].mat_new = &mat_new;
        args[i].mat_old = &mat_old;

        if(pthread_create(&thr[i], NULL, &thr_func, (void*)&args[i]))
        {
            printf("Could not create thread %d\n", i);
            return -1;
        }
    }  

    /* join here */
    for ( int i = 0; i < N_TR; ++i )
    {
        pthread_join(thr[i], NULL);
    }
	gettimeofday(&end,0);
	avg_cp_time = (double)((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec))/(double)N_IT; //has to be cast to double, otherwise the operation won't compute decimal positions
	std::cout << N << " " << N_TR << " " << avg_cp_time << " " << N_TR << std::endl;
	//std::cout << "Computed over "  << N_IT << " iterations!" << std::endl;
	//std::cout << "Average Time/Iteration: " << avg_cp_time << " us" << std::endl;
	return 0;
}
