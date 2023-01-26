#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>



// Recursive Implementation of Interleaved Pair Approach
float sumElemArrayOnCPU(float *data, int const size) {
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively
    return sumElemArrayOnCPU(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void sumElemArrayOnGPUSynced(float *g_idata, float *g_odata, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            g_idata[idx] = g_idata[idx] + g_idata[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = g_idata[idx];
    }
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
    float gpu_sum = 0;

    // allocate device memory
    float *d_idata = NULL;
    float *d_odata = NULL;
    ERR_SAFE(cudaMalloc((void **) &d_idata, bytes));
    ERR_SAFE(cudaMalloc((void **) &d_odata, grid.x * sizeof(float)));


    // cpu reduction
    iStart = time_of_day_seconds();
    float cpu_sum = sumElemArrayOnCPU(tmp, size);
    iElaps = time_of_day_seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %f\n", iElaps, cpu_sum);


  // kernel 0: warmup
    ERR_SAFE(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERR_SAFE(cudaDeviceSynchronize());
    iStart = time_of_day_seconds();
    sumElemArrayOnGPUSynced<<<grid, block>>>(d_idata, d_odata, size);
    ERR_SAFE(cudaDeviceSynchronize());
    iElaps = time_of_day_seconds() - iStart;
    ERR_SAFE(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));


    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);


    printf("Results Match? %s\n", (gpu_sum==cpu_sum)? "true" : "false");

}