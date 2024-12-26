#ifndef CULSZ_INCLUDE_CULSZ_KERNEL_H
#define CULSZ_INCLUDE_CULSZ_KERNEL_H

#include <stdint.h>
#include <cuda_runtime.h>

// cuLSZ global settings.
static const int block_per_thread = 32;

// cuSZp global settings.
static const int cmp_tblock_size = 32; // Fixed to 32, cannot be modified.
static const int dec_tblock_size = 32; // Fixed to 32, cannot be modified.
static const int cmp_chunk = 1024;
static const int dec_chunk = 1024;

// cuSZp-based function, used for conversion methods.
__global__ void cuSZp_compress_kernel_plain_f32(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_plain_f32(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

// cuSZp-based function, used for truncation methods.
__global__ void cuSZp_compress_kernel_plain_f32_truncation_cssi(const uint32_t* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const uint threshold, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_plain_f32_truncation_cssi(uint32_t* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const uint threshold, const size_t nbEle);
__global__ void cuSZp_compress_kernel_plain_f32_truncation_xpcs(const uint16_t* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const uint threshold, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_plain_f32_truncation_xpcs(uint16_t* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const uint threshold, const size_t nbEle);


// cuLSZ functions.
__global__ void cuLSZ_compression_kernel_uint32_bsize64(const uint32_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);
__global__ void cuLSZ_decompression_kernel_uint32_bsize64(uint32_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);
__global__ void cuLSZ_compression_kernel_uint16_bsize64(const uint16_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);
__global__ void cuLSZ_decompression_kernel_uint16_bsize64(uint16_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH);


#endif // CULSZ_INCLUDE_CULSZ_KERNEL_H