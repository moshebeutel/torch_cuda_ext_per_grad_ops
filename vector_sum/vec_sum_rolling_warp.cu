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

__global__ void reduceCompleteUnrollWarps8(float *g_idata, float *g_odata,
                                           unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) i_data[tid] += i_data[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) i_data[tid] += i_data[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile float *vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

__global__ void reduceUnrollWarps8(float *g_idata, float *g_odata, unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

template<unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(float *g_idata, float *g_odata,
                                     unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512) i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) i_data[tid] += i_data[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) i_data[tid] += i_data[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile float *vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
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


    // kernel 8: reduceUnrollWarps8
    ERR_SAFE(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    ERR_SAFE(cudaDeviceSynchronize());
    iStart = time_of_day_seconds();
    // reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                                                              size);
    ERR_SAFE(cudaDeviceSynchronize());
    iElaps = time_of_day_seconds() - iStart;
    ERR_SAFE(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    
    
    printf("Results Match? %s\n", (gpu_sum==cpu_sum)? "true" : "false");


}