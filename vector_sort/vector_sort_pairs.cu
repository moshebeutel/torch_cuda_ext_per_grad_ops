#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>


void printSrcSorted(float *src, float *sorted, const int N){
    for (int i = 0; i < N; i++) {
        printf("src[%i] = %f, sorted[%i] = %f\n",i ,src[i], i, sorted[i]);
    }

}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }

    return;
}

void pairSortArraysOnHost(float *src, float *sorted, const int N) {
    for (int idx = 0; idx < N - 1; idx++) {
        if(src[idx+1] > src[idx]){
            sorted[idx+1]=src[idx];
            sorted[idx]=src[idx+1];
        }else{
            sorted[idx+1]=src[idx+1];
            sorted[idx]=src[idx];
        }
    }
}

__global__ void pairSortArrayOnGPU(float *src, float *sorted, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N-1){
        if(src[idx+1] > src[idx]){
            sorted[idx+1]=src[idx];
            sorted[idx]=src[idx+1];
        }else{
            sorted[idx+1]=src[idx+1];
            sorted[idx]=src[idx];
        }
    }
}

int main() {
    // printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    ERR_SAFE(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    ERR_SAFE(cudaSetDevice(dev));

    // set up data size of vectors
    // int nElem = 1 << 24;
    int nElem = 16;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_src, *hostRefSorted, *gpuRefSorted;
    h_src = (float *) malloc(nBytes);

    hostRefSorted = (float *) malloc(nBytes);
    gpuRefSorted = (float *) malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = time_of_day_seconds();
    initialData(h_src, nElem);
    iElaps = time_of_day_seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRefSorted, 0, nBytes);
    memset(gpuRefSorted, 0, nBytes);

    // add vector at host side for result checks
    iStart = time_of_day_seconds();
    pairSortArraysOnHost(h_src, hostRefSorted, nElem);
    iElaps = time_of_day_seconds() - iStart;
    printf("pairSortArraysOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_src, *d_sorted;
    ERR_SAFE(cudaMalloc((float **) &d_src, nBytes));
    ERR_SAFE(cudaMalloc((float **) &d_sorted, nBytes));

    // transfer data from host to device
    ERR_SAFE(cudaMemcpy(d_src, h_src, nBytes, cudaMemcpyHostToDevice));
    ERR_SAFE(cudaMemcpy(d_sorted, gpuRefSorted, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = nElem;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    iStart = time_of_day_seconds();
    pairSortArrayOnGPU<<<grid, block>>>(d_src, d_sorted, nElem);
    ERR_SAFE(cudaDeviceSynchronize());
    iElaps = time_of_day_seconds() - iStart;
    printf("pairSortArrayOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    // check kernel error
    ERR_SAFE(cudaGetLastError());

    // copy kernel result back to host side
    ERR_SAFE(cudaMemcpy(gpuRefSorted, d_sorted, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRefSorted, gpuRefSorted, nElem);

    // print source vs sorted
    printf("*** Source vs Sorted Host\n");
    printSrcSorted(h_src, hostRefSorted, nElem);
    printf("*** Source vs Sorted Device\n");
    printSrcSorted(h_src, gpuRefSorted, nElem);

    // free device global memory
    ERR_SAFE(cudaFree(d_src));
    ERR_SAFE(cudaFree(d_sorted));

    // free host memory
    free(h_src);
    free(hostRefSorted);
    free(gpuRefSorted);

    return (0);
}

