#include "cuLSZ_entry.h"
#include "cuLSZ_kernel.h"

// just for debugging, remember to delete later.
#include <stdio.h> 
// Define a macro for error checking
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
// Function to check CUDA errors
void check(cudaError_t result, const char *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, (unsigned int)result, cudaGetErrorString(result));
        // Exit if there is an error
        exit(result);
    }
}


void cuSZp_compress_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_compress_kernel_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    checkCudaErrors(cudaGetLastError());

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


void cuSZp_decompress_plain_f32(float* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, float errorBound, cudaStream_t stream)
{
    // Data blocking.
    int bsize = dec_tblock_size;
    int gsize = (nbEle + bsize * dec_chunk - 1) / (bsize * dec_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU decompression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuSZp_decompress_kernel_plain_f32<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, errorBound, nbEle);
    checkCudaErrors(cudaGetLastError());

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


void cuLSZ_compression_uint32_bsize64(uint32_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuLSZ_compression_kernel_uint32_bsize64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *cmpSize = (size_t)glob_sync + blockNum;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void cuLSZ_decompression_uint32_bsize64(uint32_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuLSZ_decompression_kernel_uint32_bsize64<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void cuLSZ_compression_uint16_bsize64(uint16_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    size_t* d_cmpOffset;
    size_t* d_locOffset;
    int* d_flag;
    size_t glob_sync;
    cudaMalloc((void**)&d_cmpOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuLSZ_compression_kernel_uint16_bsize64<<<gridSize, blockSize, sizeof(size_t)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpy(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(size_t), cudaMemcpyDeviceToHost);
    *cmpSize = glob_sync + blockNum;

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}

void cuLSZ_decompression_uint16_bsize64(uint16_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingTH, cudaStream_t stream)
{
    // Data blocking.
    // Treating 3D data as a set of 2D slice, for each slice, we have 8x8 2D blocks.
    uint dimyBlock = (dims.y + 7) / 8;
    uint dimzBlock = (dims.z + 7) / 8;
    uint blockNum = dims.x * dimyBlock * dimzBlock;
    int bsize = 32; // One warp one threadblock for glob sync.
    int gsize = (blockNum + bsize * block_per_thread - 1) / (bsize * block_per_thread);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    size_t* d_cmpOffset;
    size_t* d_locOffset;
    int* d_flag;
    cudaMalloc((void**)&d_cmpOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_locOffset, sizeof(size_t)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(size_t)*cmpOffSize);
    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // Decompression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);
    cuLSZ_decompression_kernel_uint16_bsize64<<<gridSize, blockSize, sizeof(size_t)*2, stream>>>(d_decData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, blockNum, dims, quantBins, poolingTH);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Free memory that is used.
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}
