#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<cuda.h>
#include<cuda_runtime.h>

// Result from last compute of world.
// Add extern keyworld to call from C file
extern "C" {unsigned char *g_resultData = NULL;}

// Current state of world. 
extern "C" {unsigned char *g_data = NULL;}

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

// Ghost rows at the boundaries of MPI ranks
extern unsigned char *ghost_top;
extern unsigned char *ghost_bottom;

// World all zeros
static inline void gol_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

// World All ones
static inline void gol_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));

    // set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }
    
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

// Modify 10 ones at 128 columns in the last row for each rank
static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));
    
    for ( i = 0; i < g_dataLength; i++ )
    {
        if (i > ((worldWidth * (worldHeight-1)) + 126) && i <  ((worldWidth * (worldHeight-1)) + 127))
        {
            g_data[i] = 1;
        }
    }

    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

// Ones at the corners of the World
// Only modify MPI rank 0 and the last MPI rank
static inline void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));
    // rank 0 
    if (myrank == 0 )
    {
        g_data[0] = 1; // upper left
        g_data[worldWidth-1] = 1; // upper right
    }
    // last rank
    if (myrank == numranks -1 )
    {
        g_data[(worldHeight * (worldWidth-1))]=1; // lower left
        g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    }
    
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

// "Spinner" pattern at corners of the world
// Only modify rank 0
static inline void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight, int myrank, int numranks)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));
    // rank 0
    if (myrank == 0 )
    {
        g_data[0] = 1; // upper left
        g_data[1] = 1; // upper left +1
        g_data[worldWidth-1] = 1; // upper right
    }
    
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

// Add extern keyworld to call from C file 
extern "C" void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks)
{
    char filename[1024];
    FILE *fptr;
    sprintf(filename, "output-%d-%zu-%d.txt", numranks, worldWidth, myrank);

    fptr = fopen(filename, "a");
    int cE, cudaDeviceCount;
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        fprintf(fptr, " Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        fprintf(fptr, " Unable to have rank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }
    
    switch(pattern)
    {
    case 0:
	gol_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	gol_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	gol_initOnesInMiddle( worldWidth, worldHeight, myrank, numranks);
	break;
	
    case 3:
	gol_initOnesAtCorners( worldWidth, worldHeight, myrank, numranks);
	break;

    case 4:
	gol_initSpinnerAtCorner( worldWidth, worldHeight, myrank, numranks);
	break;

    default:
    fprintf(fptr, "Pattern %u has not been implemented \n", pattern);
    fclose(fptr);
	exit(-1);
    }
}

// Allocate space for ghost rows
extern "C" void gol_Malloc ()
{
    cudaMallocManaged (&ghost_top, (g_worldWidth * sizeof (unsigned char)));
    memset(ghost_top, 0, g_worldWidth);
    cudaMallocManaged (&ghost_bottom, (g_worldWidth * sizeof (unsigned char)));
    memset(ghost_bottom, 0, g_worldWidth);
}

// swap the pointers of pA and pB
static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;

}

// Define gol_countAliveCells in device code
// Count alive cells using ghost rows
__device__ static inline unsigned int gol_countAliveCells(const unsigned char* data, 
					   size_t x0, 
					   size_t x1, 
					   size_t x2, 
					   size_t y0, 
					   size_t y1,
                       size_t y2,
                       const unsigned char* ghost_top,
                       const unsigned char* ghost_bottom,
                       unsigned int worldWidth,
                       unsigned int worldHeight) 
{
    size_t index;
    size_t left;
    size_t right;
    unsigned int lives = 0;
    index = x1 + y1;
    right = (index%worldWidth +1)%worldWidth;
    left = (index%worldWidth + worldWidth -1)%worldWidth;
    // return the number of alive cell for data[x1+y1]
    // first row uses the ghost_top row to update
    if ( index <  worldWidth)
    {
        // return the number of alive cell for data[x1+y1]
        // there are 8 neighbors 
        // up left 
        lives += ghost_top[left];
        // up
        lives += ghost_top[index];
        // up right 
        lives += ghost_top[right];
        // left 
        lives += data[x0+y1];
        // right 
        lives += data[x2+y1];
        // buttom left
        lives += data[x0+y2];
        // buttom
        lives += data[x1+y2];
        // buttom right 
        lives += data[x2+y2];
    }
    // last row uses the ghost_bottom row to update
    else if( index >= worldWidth * (worldHeight -1))
    {
        // return the number of alive cell for data[x1+y1]
        // there are 8 neighbors 
        // up left 
        lives += data[x0+y0];
        // up
        lives += data[x1+y0];
        // up right 
        lives += data[x2+y0];
        // left 
        lives += data[x0+y1];
        // right 
        lives += data[x2+y1];
        // buttom left
        lives += ghost_bottom[left];
        // buttom
        lives += ghost_bottom[index%worldWidth];
        // buttom right 
        lives += ghost_bottom[right];
    }
    else
    {
        // there are 8 neighbors 
        // up left 
        lives += data[x0+y0];
        // up
        lives += data[x1+y0];
        // up right 
        lives += data[x2+y0];
        // left 
        lives += data[x0+y1];
        // right 
        lives += data[x2+y1];
        // buttom left
        lives += data[x0+y2];
        // buttom
        lives += data[x1+y2];
        // buttom right 
        lives += data[x2+y2];

    }
    // return place holder to avoid warning
    return lives;
}

// Printf my rank's chunk of universe to separate file
extern "C" void gol_printWorld(int myrank, int numranks)
{
    char filename[1024];
    FILE *fptr;
    sprintf(filename, "output-%d-%zu-%d.txt", numranks, g_worldWidth, myrank);

    fptr = fopen(filename, "a");
    int i, j;

    fprintf(fptr, "######################### FINAL WORLD IS ###############################\n");

    for( i = 0; i < g_worldHeight; i++)
    {
	    fprintf(fptr, "Row %2d: ", i);
	    for( j = 0; j < g_worldWidth; j++)
	    {
	        fprintf(fptr, "%u ", (unsigned int)g_data[(i * g_worldWidth) + j]);
	    }
	    fprintf(fptr, "\n");
    }

    fprintf(fptr, "\n\n");
    fclose(fptr);
}

// Main CUDA kernel function to compute the world
// Make changes in d_resultData array
__global__ void gol_kernel (const unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData, int myrank, int numranks, unsigned char* ghost_top, unsigned char* ghost_bottom)
{
    
    // Allocate elements to threads
    // Use blockIdx.x to access block index within grid
    // Use threadIdx.x to access thread index within block
    // Each index means the current cell ID
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; 
         index < worldWidth * worldHeight;
         index += blockDim.x * gridDim.x)
    {
        size_t y,x;
        size_t y0, y1, y2, x0, x1, x2;
        // Convert current cell ID to 2D array structure (x is the collumn number, y is the row number)
        y = index/worldWidth;
        x = index % worldWidth;
        // Compute the x0, x1, x2, y0, y1, y2 offsets using x, y, worldHeight and worldWidth variables
        y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        y1 = y * worldWidth;
        y2 = ((y + 1) % worldHeight) * worldWidth;

        x1 = x;
        x0 = (x1 + worldWidth - 1) % worldWidth;
        x2 = (x1 + 1) % worldWidth;
        // Call gol_countAliveCells function to count the alive cells from 8 neighbors using ghost rows
        unsigned int lives = gol_countAliveCells(d_data, x0, x1, x2, y0, y1, y2, ghost_top, ghost_bottom, worldWidth, worldHeight);
        // Consider all situations to compute the world
        if (d_data[index] == 1 && lives < 2)
        {
            d_resultData[x1 + y1 ] = 0;  
        }      
        if (d_data[x1 + y1 ] == 1 && lives >= 2 && lives <= 3)
        {
            d_resultData[x1 + y1 ] = 1;  
        }
        if (d_data[x1 + y1 ] == 1 && lives > 3)
        {
            d_resultData[x1 + y1 ] = 0;  
        }
        if (d_data[x1 + y1 ] == 0 && lives < 3)
        {
            d_resultData[x1 + y1 ] = 0;  
        }
        if (d_data[x1 + y1 ] == 0 && lives > 3)
        {
            d_resultData[x1 + y1 ] = 0;  
        }
        if (d_data[x1 + y1 ] == 0 && lives == 3)
        {
            d_resultData[x1 + y1 ] = 1;  
        }

    }
    
    
}

// Launch the CUDA kernel and is called from main
extern "C" bool gol_kernelLaunch (unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, ushort threadsCount, int myrank, int numranks, unsigned char* ghost_top, unsigned char* ghost_bottom)
{   
    //size_t i;
    size_t dataLength;
    dataLength = worldWidth * worldHeight;

    // Invoke the gol_kernel() CUDA kernel function on GPU
    // Launch parallel threads, kernel<<<block_number, threads_number>>>(...)
    gol_kernel<<<dataLength/threadsCount, threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData, myrank, numranks, ghost_top, ghost_bottom);
    // Swap the new world with the previous world to be ready for next iteration
    gol_swap(d_resultData, d_data);

    // Synchronize all threads within the block
    // Block the CPU until all preceding CUDA calls have completed
    cudaDeviceSynchronize();

    return true;

}



