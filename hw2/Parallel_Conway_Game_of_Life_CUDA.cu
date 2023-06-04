#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<cuda.h>
#include<cuda_runtime.h>

// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world. 
unsigned char *g_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

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

static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));

    // set first 1 rows of world to true
    for( i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
	if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
	{
	    g_data[i] = 1;
	}
    }
    
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

static inline void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));

    g_data[0] = 1; // upper left
    g_data[worldWidth-1]=1; // upper right
    g_data[(worldHeight * (worldWidth-1))]=1; // lower left
    g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

static inline void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // initialize data on the CPU
    cudaMallocManaged (&g_data, (g_dataLength * sizeof (unsigned char)));
    // zero out all the data elements 
    memset(g_data, 0, (g_dataLength * sizeof (unsigned char)));

    g_data[0] = 1; // upper left
    g_data[1] = 1; // upper left +1
    g_data[worldWidth-1]=1; // upper right
    
    cudaMallocManaged (&g_resultData, (g_dataLength * sizeof (unsigned char)));
    memset(g_resultData, 0, (g_dataLength * sizeof (unsigned char)));

}

static inline void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
    switch(pattern)
    {
    case 0:
	gol_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	gol_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	gol_initOnesInMiddle( worldWidth, worldHeight );
	break;
	
    case 3:
	gol_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	gol_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

// swap the pointers of pA and pB
static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;

}

// define gol_countAliveCells in device code
__device__ static inline unsigned int gol_countAliveCells(const unsigned char* data, 
					   size_t x0, 
					   size_t x1, 
					   size_t x2, 
					   size_t y0, 
					   size_t y1,
					   size_t y2) 
{
  
    // return the number of alive cell for data[x1+y1]
    // there are 8 neighbors 
    unsigned int lives = 0;
    lives += data[x0+y0];
    lives += data[x1+y0];
    lives += data[x2+y0];
    lives += data[x0+y1];
    lives += data[x2+y1];
    lives += data[x0+y2];
    lives += data[x1+y2];
    lives += data[x2+y2];

    // return place holder to avoid warning
    return lives;
}

static inline void gol_printWorld()
{
    int i, j;

    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}

// main CUDA kernel function to compute the world
// compute the world and make changes in d_resultData array
__global__ void gol_kernel (const unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData)
{
    
    // allocate elements to threads
    // use blockIdx.x to access block index within grid
    // use threadIdx.x to access thread index within block
    // each index means the current cell ID
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; 
         index < worldWidth * worldHeight;
         index += blockDim.x * gridDim.x)
    {
        size_t y,x;
        size_t y0, y1, y2, x0, x1, x2;
        // convert current cell ID to 2D array structure (x is the collumn number, y is the row number)
        y = index/worldWidth;
        x = index % worldWidth;
        // compute the x0, x1, x2, y0, y1, y2 offsets using x, y, worldHeight and worldWidth variables
        y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        y1 = y * worldWidth;
        y2 = ((y + 1) % worldHeight) * worldWidth;

        x1 = x;
        x0 = (x1 + worldWidth - 1) % worldWidth;
        x2 = (x1 + 1) % worldWidth;
        // call gol_countAliveCells function to count the alive cells from 8 neighbors 
        unsigned int lives = gol_countAliveCells(d_data, x0, x1, x2, y0, y1, y2);
        // consider all situations to compute the world
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
bool gol_kernelLaunch (unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount)
{   
    size_t i;
    size_t dataLength;
    dataLength = worldWidth * worldHeight;

    for(i = 0; i < iterationsCount; ++i)
    {
        // Invoke the gol_kernel() CUDA kernel function on GPU
        // Launch parallel threads, kernel<<<block_number, threads_number>>>(...)
        gol_kernel<<<dataLength/threadsCount, threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData);
        // Swap the new world with the previous world to be ready for next iteration
        gol_swap(d_resultData, d_data);
    }


    // Synchronize all threads within the block
    // Block the CPU until all preceding CUDA calls have completed
    cudaDeviceSynchronize();

    return true;

}




int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    unsigned int threads = 0;
    bool output;

    printf("This is the Game of Life running in parallel on a GPU.\n");

    if( argc != 6 )
    {
	printf("GOL requires 5 arguments: pattern number, sq size of the world, the number of itterations, the number of threads per block and output pattern e.g. ./gol 4 64 2 2 0\n");
	exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    threads = atoi(argv[4]);
    // boolean type variable to control the output is on or off
    output =  atoi(argv[5]);

    gol_initMaster(pattern, worldSize, worldSize);
    gol_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, itterations, threads);

    // if the value is 1, then output is turned on
    // if the value is 0, then output is turned off
    if(output)
    {
        printf("######################### FINAL WORLD IS ###############################\n");
        gol_printWorld();
    }
    
    // free the memory we allocated on the GPU and CPU
    cudaFree(g_data);
    cudaFree(g_resultData);

    return true;
}
