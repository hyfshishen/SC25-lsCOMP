#ifndef CULSZ_INCLUDE_CULSZ_ENTRY_H
#define CULSZ_INCLUDE_CULSZ_ENTRY_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

void cuLSZ_compression_uint32_bsize64(uint32_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void cuLSZ_decompression_uint32_bsize64(uint32_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void cuLSZ_compression_uint16_bsize64(uint16_t* d_oriData, unsigned char* d_cmpBytes, size_t* cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);
void cuLSZ_decompression_uint16_bsize64(uint16_t* d_decData, unsigned char* d_cmpBytes, size_t cmpSize, uint3 dims, uint4 quantBins, float poolingSH, cudaStream_t stream=0);

#endif // CULSZ_INCLUDE_CULSZ_ENTRY_H