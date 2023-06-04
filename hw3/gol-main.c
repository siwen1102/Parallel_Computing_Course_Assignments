#include <stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<mpi.h>


// Global variable 
// Result from last compute of world.
extern unsigned char *g_resultData;
// Current state of world. 
extern unsigned char *g_data;
// Ghost rows at the boundaries of MPI ranks
unsigned char *ghost_top = NULL;
unsigned char *ghost_bottom = NULL;

extern void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks); 
extern void gol_Malloc ();
extern bool gol_kernelLaunch (unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, ushort threadsCount, int myrank, int numranks, unsigned char* ghost_top, unsigned char* ghost_bottom);
extern void gol_printWorld( int myrank, int numranks);

int main(int argc, char** argv) {
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    unsigned int threads = 0;
    // output pattern 
    bool output;

    MPI_Request request1, request2, request3, request4;
    MPI_Status status1, status2;
    int myrank, numranks, up, down;
    double t1, t2;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the rank of process
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    printf("This is the Game of Life running in parallel in CUDA and MPI.\n");

    if( argc != 6 )
    {
        printf("GOL requires 5 arguments: pattern number, sq size of the world, the number of itterations, the number of threads per block and output pattern e.g. ./gol-cuda-mpi-exe 4 64 2 2 0\n");
        exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    threads = atoi(argv[4]);
    output =  atoi(argv[5]);

    // Set CUDA Device based on MPI rank
    gol_initMaster(pattern, worldSize, worldSize, myrank, numranks);

    if(myrank == 0){
        // start time
        t1 = MPI_Wtime();
    }

    // Allocate space for ghost rows 
    gol_Malloc();
    gol_Malloc();

    // up is the previous rank id 
    // down is the next rank id 
    up = (myrank + numranks - 1)%numranks;
    down = (myrank+1)%numranks;

    for( int i = 0; i < itterations; i++)
    {
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv
        // Recv row data from the up rank
        // Save row data to ghost_top row
        MPI_Irecv(
            /* to receive    =*/ ghost_top,
            /* count         =*/ worldSize,
            /* datatype      =*/ MPI_UNSIGNED_CHAR,
            /* source        =*/ up,
            /* tag           =*/ 0,
            /* communicator  =*/ MPI_COMM_WORLD,
            /* request       =*/ &request1); 

        // Recv row data from the down rank
        // Save row data to ghost_bottom row
        MPI_Irecv(
            /* to receive    =*/ ghost_bottom,
            /* count         =*/ worldSize,
            /* datatype      =*/ MPI_UNSIGNED_CHAR,
            /* source        =*/ down,
            /* tag           =*/ 1,
            /* communicator  =*/ MPI_COMM_WORLD,
            /* request       =*/ &request2); 

        // Send the last row data to the down rank
        MPI_Isend(
            /* to send       =*/ g_data+(worldSize*(worldSize-1)),
            /* count         =*/ worldSize,
            /* datatype      =*/ MPI_UNSIGNED_CHAR,
            /* destination   =*/ down,
            /* tag           =*/    0,
            /* communicator  =*/ MPI_COMM_WORLD,
            /* request       =*/ &request3);

        // Send the first row data to the up rank
        MPI_Isend(
            /* to send       =*/ g_data,
            /* count         =*/ worldSize,
            /* datatype      =*/ MPI_UNSIGNED_CHAR,
            /* destination   =*/ up,
            /* tag           =*/    1,
            /* communicator  =*/ MPI_COMM_WORLD,
            /* request       =*/ &request4);

        // Synchronize 
        MPI_Wait(&request1, &status1);
        MPI_Wait(&request2, &status2);

        // Do rest of universe update using CUDA GOL kernel
        gol_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, threads, myrank, numranks, ghost_top, ghost_bottom);
        
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(myrank == 0 ){
        // end time 
        t2 = MPI_Wtime();
        // printf MPI_Wtime performance results
        printf("MPI Wtime is %f\n", t2-t1);
    }
    // if output argument is true
    // print my Rank's chunk of universe to separate file  
    if(output)
    {
        gol_printWorld( myrank, numranks);
    }

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
