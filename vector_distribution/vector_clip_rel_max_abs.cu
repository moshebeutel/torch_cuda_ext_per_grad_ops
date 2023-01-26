#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#define MAX_CUDA_THREADS_PER_BLOCK 1024



void clip_rel_max_abs_cpu(float *data, const float clip_frac,int const size) {
    auto max_abs = 0.0f;
    for (int i = 1; i < size-1; i++) {
        max_abs = abs(data[i]) > max_abs ? abs(data[i]) : max_abs ;
    }
    auto clip_threshold = clip_frac * max_abs;
    for (int i = 1; i < size-1; i++) {
        data[i] = abs(data[i]) > clip_threshold ? clip_threshold : data[i] ;
    }

}



__global__ void clip_rel_max_abs_gpu(float* data, float* clipped_data, const float clip_frac, const int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        clipped_data[threadIdx.x] = data[idx];
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
    if (idx == 0){
        data[0] = sdata[0];
        __syncthreads();

    }

    auto clip_threshold = abs(sdata[0]) * clip_frac;

    // clip
    
    clipped_data[idx] = clipped_data[idx] > clip_threshold ? clip_threshold : clipped_data[idx]; 

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
    auto size = 1 << 16; // total number of elements 
    auto clip_frac = 0.8f;
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 1024;   // initial block size

    if (argc > 1) {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *h_idata = (float *) malloc(bytes);
    float *tmp = (float *) malloc(bytes);

    // initialize the array
    printf("Init\n");
    for (unsigned int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (float) (rand() & 0xFF);
        // printf("%f, ",h_idata[i]);
    }
    printf("\n");

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;

    // allocate device memory
    float *d_idata = NULL;
    float *d_odata = NULL;
    ERR_SAFE(cudaMalloc((void **) &d_idata, bytes));
    ERR_SAFE(cudaMalloc((void **) &d_odata, bytes));



    // cpu reduction
    iStart = time_of_day_seconds();
    clip_rel_max_abs_cpu(tmp, clip_frac, size);
    iElaps = time_of_day_seconds() - iStart;
    printf("cpu clip: elapsed %f\n", iElaps);
    // for(int i=0; i < size; i++) {
    //     printf("%f, ", tmp[i]);
    // }
    printf("\n");


  // kernel 1: interleaved addressing shared
    ERR_SAFE(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERR_SAFE(cudaMemcpy(d_odata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERR_SAFE(cudaDeviceSynchronize());
    iStart = time_of_day_seconds();
    clip_rel_max_abs_gpu<<<grid, block>>>(d_idata, d_odata, clip_frac, size);
    ERR_SAFE(cudaDeviceSynchronize());
    iElaps = time_of_day_seconds() - iStart;


//   // kernel 2: interleaved addressing global
//     ERR_SAFE(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//     ERR_SAFE(cudaDeviceSynchronize());
//     iStart = time_of_day_seconds();
//     MaxAbs_Interleaved_Addressing_Global<<<grid, block>>>(d_idata, size);
//     ERR_SAFE(cudaDeviceSynchronize());
//     iElaps = time_of_day_seconds() - iStart;

    float *gpu_clip_ref;
    gpu_clip_ref = (float *) malloc(bytes);
    ERR_SAFE(cudaMemcpy(gpu_clip_ref, d_odata, bytes, cudaMemcpyDeviceToHost));
    printf("gpu clip: elapsed %f\n", iElaps);
    // for(int i=0; i < size; i++) {
    //     printf("%f, ", gpu_clip_ref[i]);
    // }
    printf("\n");



    printf("Check clip match:\n");
   
    unsigned int i=0;
    while (gpu_clip_ref[i]==tmp[i] && ++i < size);

    // printf("Results Match? %s\n", match? "true" : "false");
    printf("i = %d",i);
    
    ERR_SAFE(cudaFree(d_idata));
    ERR_SAFE(cudaFree(d_odata));

    free(h_idata);
    free(gpu_clip_ref);

}