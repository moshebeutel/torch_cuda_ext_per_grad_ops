#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>


void printSrcSum(float *v, float *sum, const int N){
    for (int i = 0; i < N; i++) {
        printf("v[%i] = %f\n",i ,v[i]);
    }
    printf("v.sum() = %f\n", *sum);
}
void checkResult(float *hostRef, float *gpuRef) {
    printf("Results on Host Matches to Result on Device? %s", (*hostRef) == (*gpuRef) ? "true" : "false");
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

void sumElemArrayOnHost(float *v, float *hostRefSum, const int N) {
    for (int idx = 0; idx < N - 1; idx++) {
        (*hostRefSum)+=v[idx];
    }
}

__global__ void sumElemArrayOnGPU(float *v, const int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        v[0]+=v[idx];
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
    int nElem = 4;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_v, *hostRefSum, *gpuRefSum;
    h_v = (float *) malloc(nBytes);

    hostRefSum = (float *) malloc(sizeof(float));
    gpuRefSum= (float *) malloc(sizeof(float));

    double iStart, iElaps;

    // initialize data at host side
    iStart = time_of_day_seconds();
    initialData(h_v, nElem);
    iElaps = time_of_day_seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRefSum, 0, nBytes);
    memset(gpuRefSum, 0, nBytes);
    printf("Host memory allocated %d bytes\n", (int)nBytes);

    // add vector at host side for result checks
    iStart = time_of_day_seconds();
    sumElemArrayOnHost(h_v, hostRefSum, nElem);
    iElaps = time_of_day_seconds() - iStart;
    printf("sumElemArrayOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_v;
    printf("Device memory shall allocate %d bytes\n", (int)(nBytes));
    ERR_SAFE(cudaMalloc(&d_v, nBytes));
    // CHECK(cudaMalloc((float **) &d_v, nBytes));
    // CHECK(cudaMalloc((float **) &d_sum, sizeof(float)));
    printf("Device memory allocated\n");


    // transfer data from host to device
    ERR_SAFE(cudaMemcpy(d_v, h_v, nBytes, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_sum, gpuRefSum, sizeof(float), cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = nElem;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    iStart = time_of_day_seconds();
    sumElemArrayOnGPU<<<grid, block>>>(d_v, nElem);
    ERR_SAFE(cudaDeviceSynchronize());
    iElaps = time_of_day_seconds() - iStart;
    printf("sumElemArrayOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    // check kernel error
    ERR_SAFE(cudaGetLastError());

    // copy kernel result back to host side
    ERR_SAFE(cudaMemcpy(gpuRefSum, d_v, sizeof(float), cudaMemcpyDeviceToHost));


    // print host result vs device result
    printf("\n\n\n");
    printf("Vector.sum() Host\n");
    printf("******************\n");
    printSrcSum(h_v, hostRefSum, nElem);
    printf("\n\n\n");
    printf("Vector.sum() Device\n");
    printf("******************\n");
    printSrcSum(h_v, gpuRefSum, nElem);
    printf("\n\n\n");
    // check results
    checkResult(hostRefSum, gpuRefSum);
    printf("\n\n\n");
    // free device global memory
    ERR_SAFE(cudaFree(d_v));
    // CHECK(cudaFree(d_sum));

    // free host memory
    free(h_v);
    free(hostRefSum);
    free(gpuRefSum);

    return (0);
}

