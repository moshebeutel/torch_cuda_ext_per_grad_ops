#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#define MAX_CUDA_THREADS_PER_BLOCK 1024



void maxabsElemArrayOnCPU(float *data, int const size) {
    // in-place reduction
    for (int i = 1; i < size-1; i++) {
        data[0] = abs(data[i]) > abs(data[0]) ? data[i] : data[0] ;
    }
}

__global__ void MaxAbs_Interleaved_Addressing_Global(float* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size){
        for(int stride=1; stride < data_size; stride *= 2) {
            if (idx % (2*stride) == 0) {
                float lhs = data[idx];
                float rhs = data[idx + stride];
                data[idx] = abs(lhs) < abs(rhs) ? rhs : lhs;
            }
            __syncthreads();
        }
    }
}

__global__ void MaxAbs_Interleaved_Addressing_Shared(float* data, const int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=1; stride < blockDim.x; stride *= 2) {
            if (threadIdx.x % (2*stride) == 0) {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = abs(lhs) < abs(rhs) ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}


int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    ERR_SAFE(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    ERR_SAFE(cudaSetDevice(dev));

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if (argc > 1) {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *h_idata = (float *) malloc(bytes);
    float *h_odata = (float *) malloc(grid.x * sizeof(float));
    float *tmp = (float *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (float) (rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;

    // allocate device memory
    float *d_idata = NULL;
    ERR_SAFE(cudaMalloc((void **) &d_idata, bytes));



    // cpu reduction
    iStart = time_of_day_seconds();
    maxabsElemArrayOnCPU(tmp, size);
    auto cpu_maxabs = tmp[0];
    iElaps = time_of_day_seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_max: %f\n", iElaps, cpu_maxabs);


  // kernel 1: interleaved addressing shared
    ERR_SAFE(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERR_SAFE(cudaDeviceSynchronize());
    iStart = time_of_day_seconds();
    MaxAbs_Interleaved_Addressing_Shared<<<grid, block>>>(d_idata, size);
    ERR_SAFE(cudaDeviceSynchronize());
    iElaps = time_of_day_seconds() - iStart;


//   // kernel 2: interleaved addressing global
//     ERR_SAFE(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//     ERR_SAFE(cudaDeviceSynchronize());
//     iStart = time_of_day_seconds();
//     MaxAbs_Interleaved_Addressing_Global<<<grid, block>>>(d_idata, size);
//     ERR_SAFE(cudaDeviceSynchronize());
//     iElaps = time_of_day_seconds() - iStart;

    float gpu_maxabs;
    ERR_SAFE(cudaMemcpy(&gpu_maxabs, d_idata, sizeof(float), cudaMemcpyDeviceToHost));
    printf("gpu reduce      elapsed %f sec gpu_max: %f\n", iElaps, gpu_maxabs);

    ERR_SAFE(cudaFree(d_idata));

    printf("Results Match? %s\n", (gpu_maxabs==cpu_maxabs)? "true" : "false");

}