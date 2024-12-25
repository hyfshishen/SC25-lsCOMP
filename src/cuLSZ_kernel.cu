#include "cuLSZ_kernel.h"
#include <stdio.h> // just for debugging, remember to delete later.

__device__ inline int quantization(float data, float recipPrecision)
{
    int result;
    asm("{\n\t"
        ".reg .f32 dataRecip;\n\t"
        ".reg .f32 temp1;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dataRecip, %1, %2;\n\t"
        "setp.ge.f32 p, dataRecip, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 temp1, dataRecip, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, temp1;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}": "=r"(result) : "f"(data), "f"(recipPrecision)
    );
    return result;
}


__device__ inline int get_bit_num(unsigned int x)
{
    int leading_zeros;
    asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(x));
    return 32 - leading_zeros;
}


__global__ void cuSZp_compress_kernel_plain_f32(const float* const __restrict__ oriData, 
                                                unsigned char* const __restrict__ cmpData, 
                                                volatile unsigned int* const __restrict__ cmpOffset, 
                                                volatile unsigned int* const __restrict__ locOffset, 
                                                volatile int* const __restrict__ flag, 
                                                const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk >> 5;
    const int rate_ofs = (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
    const float recipPrecision = 0.5f/eb;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[cmp_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0; // Thread-level prefix-sum, double check for overflow in large data (can be resolved by using size_t type).
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * cmp_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        block_idx = base_block_start_idx/32;
        prevQuant = 0;
        maxQuant = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i=base_block_start_idx; i<base_block_end_idx; i+=4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i/4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = i % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+1) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+2) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+3) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i=base_block_start_idx; i<base_block_start_idx+remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_ofs = i % 32;
                    sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int chunk_idx_start = j*32;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char.x = 0xff & (sign_flag[j] >> 24);
            tmp_char.y = 0xff & (sign_flag[j] >> 16);
            tmp_char.z = 0xff & (sign_flag[j] >> 8);
            tmp_char.w = 0xff & sign_flag[j];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
            cmp_byte_ofs+=4;

            int mask = 1;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;

                tmp_char.x = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_plain_f32(float* const __restrict__ decData, 
                                                  const unsigned char* const __restrict__ cmpData, 
                                                  volatile unsigned int* const __restrict__ cmpOffset, 
                                                  volatile unsigned int* const __restrict__ locOffset, 
                                                  volatile int* const __restrict__ flag, 
                                                  const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = dec_chunk >> 5;
    const int rate_ofs = (nbEle+dec_tblock_size*dec_chunk-1)/(dec_tblock_size*dec_chunk)*(dec_tblock_size*dec_chunk)/32;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * dec_chunk + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * dec_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        unsigned int sign_flag = 0;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
            sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                        (0x00ff0000 & (tmp_char.y << 16)) |
                        (0x0000ff00 & (tmp_char.z << 8))  |
                        (0x000000ff & tmp_char.w);
            cmp_byte_ofs+=4;
            
            for(int i=0; i<32; i++) absQuant[i] = 0;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;

                absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
            }
            
            prevQuant = 0;
            if(base_block_end_idx < nbEle)
            {
                #pragma unroll 8
                for(int i=0; i<32; i+=4)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    sign_ofs = (i+1) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    sign_ofs = (i+2) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    sign_ofs = (i+3) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;
                    
                    reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                }
            }
            else
            {
                for(int i=0; i<32; i++)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                }
            }      
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}



__global__ void cuLSZ_compression_kernel_uint32_bsize64(const uint32_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    unsigned int thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.
    
    // Vector-quantization, Dynamic Binning Selection, Fixed-length Encoding.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }
            
            // Preparation for ratio profiling.
            uint zero_count = 0;
            uint zero_count_bins[4] = {0, 0, 0, 0};
            uint max_val1 = 0;
            uint max_val2 = 0;
            for(int i=0; i<64; i++)
            {
                uint val = block_data[i];
                zero_count += (val == 0);
                zero_count_bins[0] += (val < quant_bins[0]); // Base bin operation
                zero_count_bins[1] += (val < quant_bins[1]);
                zero_count_bins[2] += (val < quant_bins[2]);
                zero_count_bins[3] += (val < quant_bins[3]);
                max_val1 = (val > max_val1) ? val : max_val1;
                if(i%2)
                {
                    uint tmp_val = (block_data[i-1] + block_data[i]) / 2;
                    max_val2 = (tmp_val > max_val2) ? tmp_val : max_val2;
                }
            }

            // Compression algorithm selection and store meta data.
            float sparsity = (float)zero_count / 64;
            int pooling_choice = (sparsity > poolingTH);
            uint bin_choice = 0;
            // Progressively bin size selection.
            if(zero_count_bins[1]==zero_count_bins[0])
            {
                bin_choice = 1;
                if(zero_count_bins[2]==zero_count_bins[1])
                {
                    bin_choice = 2;
                    if(zero_count_bins[3]==zero_count_bins[2])
                    {
                        bin_choice = 3;
                    }
                }
            }

            // Store meta data.
            int max_quantized_val;
            int temp_rate = 0;
            if(pooling_choice)
            {
                max_quantized_val = max_val2 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 4;
                temp_rate = 0x80 | (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
            else
            {
                max_quantized_val = max_val1 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 8;
                temp_rate = (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also storing data to global memory.
    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }

            // Retrieve meta data.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;
            
            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                if(pooling_choice)
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) 
                    {
                        pooling_block_data[i] = (block_data[i*2] + block_data[i*2+1]) / 2;
                        pooling_block_data[i] = pooling_block_data[i] / quant_bins[bin_choice];
                    }

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer.x = 0;
                        tmp_buffer.y = 0;
                        tmp_buffer.z = 0;
                        tmp_buffer.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer.x = (((pooling_block_data[0] & mask) >> i) << 7) |
                                       (((pooling_block_data[1] & mask) >> i) << 6) |
                                       (((pooling_block_data[2] & mask) >> i) << 5) |
                                       (((pooling_block_data[3] & mask) >> i) << 4) |
                                       (((pooling_block_data[4] & mask) >> i) << 3) |
                                       (((pooling_block_data[5] & mask) >> i) << 2) |
                                       (((pooling_block_data[6] & mask) >> i) << 1) |
                                       (((pooling_block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer.y = (((pooling_block_data[8] & mask) >> i) << 7) |
                                       (((pooling_block_data[9] & mask) >> i) << 6) |
                                       (((pooling_block_data[10] & mask) >> i) << 5) |
                                       (((pooling_block_data[11] & mask) >> i) << 4) |
                                       (((pooling_block_data[12] & mask) >> i) << 3) |
                                       (((pooling_block_data[13] & mask) >> i) << 2) |
                                       (((pooling_block_data[14] & mask) >> i) << 1) |
                                       (((pooling_block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer.z = (((pooling_block_data[16] & mask) >> i) << 7) |
                                       (((pooling_block_data[17] & mask) >> i) << 6) |
                                       (((pooling_block_data[18] & mask) >> i) << 5) |
                                       (((pooling_block_data[19] & mask) >> i) << 4) |
                                       (((pooling_block_data[20] & mask) >> i) << 3) |
                                       (((pooling_block_data[21] & mask) >> i) << 2) |
                                       (((pooling_block_data[22] & mask) >> i) << 1) |
                                       (((pooling_block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer.w = (((pooling_block_data[24] & mask) >> i) << 7) |
                                       (((pooling_block_data[25] & mask) >> i) << 6) |
                                       (((pooling_block_data[26] & mask) >> i) << 5) |
                                       (((pooling_block_data[27] & mask) >> i) << 4) |
                                       (((pooling_block_data[28] & mask) >> i) << 3) |
                                       (((pooling_block_data[29] & mask) >> i) << 2) |
                                       (((pooling_block_data[30] & mask) >> i) << 1) |
                                       (((pooling_block_data[31] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer;
                        cmp_byte_ofs += 4;
                        mask <<= 1;  
                    }
                }
                else
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] / quant_bins[bin_choice];

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer1.x = 0;
                        tmp_buffer1.y = 0;
                        tmp_buffer1.z = 0;
                        tmp_buffer1.w = 0;
                        tmp_buffer2.x = 0;
                        tmp_buffer2.y = 0;
                        tmp_buffer2.z = 0;
                        tmp_buffer2.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer1.x = (((block_data[0] & mask) >> i) << 7) |
                                        (((block_data[1] & mask) >> i) << 6) |
                                        (((block_data[2] & mask) >> i) << 5) |
                                        (((block_data[3] & mask) >> i) << 4) |
                                        (((block_data[4] & mask) >> i) << 3) |
                                        (((block_data[5] & mask) >> i) << 2) |
                                        (((block_data[6] & mask) >> i) << 1) |
                                        (((block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer1.y = (((block_data[8] & mask) >> i) << 7) |
                                        (((block_data[9] & mask) >> i) << 6) |
                                        (((block_data[10] & mask) >> i) << 5) |
                                        (((block_data[11] & mask) >> i) << 4) |
                                        (((block_data[12] & mask) >> i) << 3) |
                                        (((block_data[13] & mask) >> i) << 2) |
                                        (((block_data[14] & mask) >> i) << 1) |
                                        (((block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer1.z = (((block_data[16] & mask) >> i) << 7) |
                                        (((block_data[17] & mask) >> i) << 6) |
                                        (((block_data[18] & mask) >> i) << 5) |
                                        (((block_data[19] & mask) >> i) << 4) |
                                        (((block_data[20] & mask) >> i) << 3) |
                                        (((block_data[21] & mask) >> i) << 2) |
                                        (((block_data[22] & mask) >> i) << 1) |
                                        (((block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer1.w = (((block_data[24] & mask) >> i) << 7) |
                                        (((block_data[25] & mask) >> i) << 6) |
                                        (((block_data[26] & mask) >> i) << 5) |
                                        (((block_data[27] & mask) >> i) << 4) |
                                        (((block_data[28] & mask) >> i) << 3) |
                                        (((block_data[29] & mask) >> i) << 2) |
                                        (((block_data[30] & mask) >> i) << 1) |
                                        (((block_data[31] & mask) >> i) << 0); 
                        
                        // Get i-th bit in 32~39 data.
                        tmp_buffer2.x = (((block_data[32] & mask) >> i) << 7) |
                                        (((block_data[33] & mask) >> i) << 6) |
                                        (((block_data[34] & mask) >> i) << 5) |
                                        (((block_data[35] & mask) >> i) << 4) |
                                        (((block_data[36] & mask) >> i) << 3) |
                                        (((block_data[37] & mask) >> i) << 2) |
                                        (((block_data[38] & mask) >> i) << 1) |
                                        (((block_data[39] & mask) >> i) << 0);
                        
                        // Get i-th bit in 40~47 data.
                        tmp_buffer2.y = (((block_data[40] & mask) >> i) << 7) |
                                        (((block_data[41] & mask) >> i) << 6) |
                                        (((block_data[42] & mask) >> i) << 5) |
                                        (((block_data[43] & mask) >> i) << 4) |
                                        (((block_data[44] & mask) >> i) << 3) |
                                        (((block_data[45] & mask) >> i) << 2) |
                                        (((block_data[46] & mask) >> i) << 1) |
                                        (((block_data[47] & mask) >> i) << 0);

                        // Get i-th bit in 48~55 data.
                        tmp_buffer2.z = (((block_data[48] & mask) >> i) << 7) |
                                        (((block_data[49] & mask) >> i) << 6) |
                                        (((block_data[50] & mask) >> i) << 5) |
                                        (((block_data[51] & mask) >> i) << 4) |
                                        (((block_data[52] & mask) >> i) << 3) |
                                        (((block_data[53] & mask) >> i) << 2) |
                                        (((block_data[54] & mask) >> i) << 1) |
                                        (((block_data[55] & mask) >> i) << 0);

                        // Get i-th bit in 56~63 data.
                        tmp_buffer2.w = (((block_data[56] & mask) >> i) << 7) |
                                        (((block_data[57] & mask) >> i) << 6) |
                                        (((block_data[58] & mask) >> i) << 5) |
                                        (((block_data[59] & mask) >> i) << 4) |
                                        (((block_data[60] & mask) >> i) << 3) |
                                        (((block_data[61] & mask) >> i) << 2) |
                                        (((block_data[62] & mask) >> i) << 1) |
                                        (((block_data[63] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer1;
                        cmp_byte_ofs += 4;
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer2;
                        cmp_byte_ofs += 4;
                        mask <<= 1; 
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}


__global__ void cuLSZ_decompression_kernel_uint32_bsize64(uint32_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile unsigned int* const __restrict__ cmpOffset, 
                                                        volatile unsigned int* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    unsigned int thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.

    // Obtain fixed-rate information for each block.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Obtain block meta data.
            fixed_rate[j] = cmpBytes[block_idx];

            // Check if pooling.
            int pooling_choice = fixed_rate[j] >> 7;
            int temp_rate = fixed_rate[j] & 0x1f;
            if(pooling_choice) thread_ofs += temp_rate * 4;
            else thread_ofs += temp_rate * 8;
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also reading data from global memory.
    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;
    
        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Initialization, guiding decoding process.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;

            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                // Buffering decompressed block data.
                uint block_data[64];

                // Read data and shuffle it back from global memory.
                if(pooling_choice)
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) pooling_block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        pooling_block_data[0] |= ((tmp_buffer.x >> 7) & 0x00000001) << i;
                        pooling_block_data[1] |= ((tmp_buffer.x >> 6) & 0x00000001) << i;
                        pooling_block_data[2] |= ((tmp_buffer.x >> 5) & 0x00000001) << i;
                        pooling_block_data[3] |= ((tmp_buffer.x >> 4) & 0x00000001) << i;
                        pooling_block_data[4] |= ((tmp_buffer.x >> 3) & 0x00000001) << i;
                        pooling_block_data[5] |= ((tmp_buffer.x >> 2) & 0x00000001) << i;
                        pooling_block_data[6] |= ((tmp_buffer.x >> 1) & 0x00000001) << i;
                        pooling_block_data[7] |= ((tmp_buffer.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        pooling_block_data[8] |= ((tmp_buffer.y >> 7) & 0x00000001) << i;
                        pooling_block_data[9] |= ((tmp_buffer.y >> 6) & 0x00000001) << i;
                        pooling_block_data[10] |= ((tmp_buffer.y >> 5) & 0x00000001) << i;
                        pooling_block_data[11] |= ((tmp_buffer.y >> 4) & 0x00000001) << i;
                        pooling_block_data[12] |= ((tmp_buffer.y >> 3) & 0x00000001) << i;
                        pooling_block_data[13] |= ((tmp_buffer.y >> 2) & 0x00000001) << i;
                        pooling_block_data[14] |= ((tmp_buffer.y >> 1) & 0x00000001) << i;
                        pooling_block_data[15] |= ((tmp_buffer.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        pooling_block_data[16] |= ((tmp_buffer.z >> 7) & 0x00000001) << i;
                        pooling_block_data[17] |= ((tmp_buffer.z >> 6) & 0x00000001) << i;
                        pooling_block_data[18] |= ((tmp_buffer.z >> 5) & 0x00000001) << i;
                        pooling_block_data[19] |= ((tmp_buffer.z >> 4) & 0x00000001) << i;
                        pooling_block_data[20] |= ((tmp_buffer.z >> 3) & 0x00000001) << i;
                        pooling_block_data[21] |= ((tmp_buffer.z >> 2) & 0x00000001) << i;
                        pooling_block_data[22] |= ((tmp_buffer.z >> 1) & 0x00000001) << i;
                        pooling_block_data[23] |= ((tmp_buffer.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        pooling_block_data[24] |= ((tmp_buffer.w >> 7) & 0x00000001) << i;
                        pooling_block_data[25] |= ((tmp_buffer.w >> 6) & 0x00000001) << i;
                        pooling_block_data[26] |= ((tmp_buffer.w >> 5) & 0x00000001) << i;
                        pooling_block_data[27] |= ((tmp_buffer.w >> 4) & 0x00000001) << i;
                        pooling_block_data[28] |= ((tmp_buffer.w >> 3) & 0x00000001) << i;
                        pooling_block_data[29] |= ((tmp_buffer.w >> 2) & 0x00000001) << i;
                        pooling_block_data[30] |= ((tmp_buffer.w >> 1) & 0x00000001) << i;
                        pooling_block_data[31] |= ((tmp_buffer.w >> 0) & 0x00000001) << i;
                    }

                    // Assign data back to block data.
                    for(int i=0; i<32; i++)
                    {
                        block_data[i*2] = pooling_block_data[i] * quant_bins[bin_choice];
                        block_data[i*2+1] = block_data[i*2];
                    }
                }
                else
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer1 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;
                        tmp_buffer2 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        block_data[0] |= ((tmp_buffer1.x >> 7) & 0x00000001) << i;
                        block_data[1] |= ((tmp_buffer1.x >> 6) & 0x00000001) << i;
                        block_data[2] |= ((tmp_buffer1.x >> 5) & 0x00000001) << i;
                        block_data[3] |= ((tmp_buffer1.x >> 4) & 0x00000001) << i;
                        block_data[4] |= ((tmp_buffer1.x >> 3) & 0x00000001) << i;
                        block_data[5] |= ((tmp_buffer1.x >> 2) & 0x00000001) << i;
                        block_data[6] |= ((tmp_buffer1.x >> 1) & 0x00000001) << i;
                        block_data[7] |= ((tmp_buffer1.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        block_data[8] |= ((tmp_buffer1.y >> 7) & 0x00000001) << i;
                        block_data[9] |= ((tmp_buffer1.y >> 6) & 0x00000001) << i;
                        block_data[10] |= ((tmp_buffer1.y >> 5) & 0x00000001) << i;
                        block_data[11] |= ((tmp_buffer1.y >> 4) & 0x00000001) << i;
                        block_data[12] |= ((tmp_buffer1.y >> 3) & 0x00000001) << i;
                        block_data[13] |= ((tmp_buffer1.y >> 2) & 0x00000001) << i;
                        block_data[14] |= ((tmp_buffer1.y >> 1) & 0x00000001) << i;
                        block_data[15] |= ((tmp_buffer1.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        block_data[16] |= ((tmp_buffer1.z >> 7) & 0x00000001) << i;
                        block_data[17] |= ((tmp_buffer1.z >> 6) & 0x00000001) << i;
                        block_data[18] |= ((tmp_buffer1.z >> 5) & 0x00000001) << i;
                        block_data[19] |= ((tmp_buffer1.z >> 4) & 0x00000001) << i;
                        block_data[20] |= ((tmp_buffer1.z >> 3) & 0x00000001) << i;
                        block_data[21] |= ((tmp_buffer1.z >> 2) & 0x00000001) << i;
                        block_data[22] |= ((tmp_buffer1.z >> 1) & 0x00000001) << i;
                        block_data[23] |= ((tmp_buffer1.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        block_data[24] |= ((tmp_buffer1.w >> 7) & 0x00000001) << i;
                        block_data[25] |= ((tmp_buffer1.w >> 6) & 0x00000001) << i;
                        block_data[26] |= ((tmp_buffer1.w >> 5) & 0x00000001) << i;
                        block_data[27] |= ((tmp_buffer1.w >> 4) & 0x00000001) << i;
                        block_data[28] |= ((tmp_buffer1.w >> 3) & 0x00000001) << i;
                        block_data[29] |= ((tmp_buffer1.w >> 2) & 0x00000001) << i;
                        block_data[30] |= ((tmp_buffer1.w >> 1) & 0x00000001) << i;
                        block_data[31] |= ((tmp_buffer1.w >> 0) & 0x00000001) << i;

                        // Get ith bit in 32~39 abs quant from global memory.
                        block_data[32] |= ((tmp_buffer2.x >> 7) & 0x00000001) << i;
                        block_data[33] |= ((tmp_buffer2.x >> 6) & 0x00000001) << i;
                        block_data[34] |= ((tmp_buffer2.x >> 5) & 0x00000001) << i;
                        block_data[35] |= ((tmp_buffer2.x >> 4) & 0x00000001) << i;
                        block_data[36] |= ((tmp_buffer2.x >> 3) & 0x00000001) << i;
                        block_data[37] |= ((tmp_buffer2.x >> 2) & 0x00000001) << i;
                        block_data[38] |= ((tmp_buffer2.x >> 1) & 0x00000001) << i;
                        block_data[39] |= ((tmp_buffer2.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 40~47 abs quant from global memory.
                        block_data[40] |= ((tmp_buffer2.y >> 7) & 0x00000001) << i;
                        block_data[41] |= ((tmp_buffer2.y >> 6) & 0x00000001) << i;
                        block_data[42] |= ((tmp_buffer2.y >> 5) & 0x00000001) << i;
                        block_data[43] |= ((tmp_buffer2.y >> 4) & 0x00000001) << i;
                        block_data[44] |= ((tmp_buffer2.y >> 3) & 0x00000001) << i;
                        block_data[45] |= ((tmp_buffer2.y >> 2) & 0x00000001) << i;
                        block_data[46] |= ((tmp_buffer2.y >> 1) & 0x00000001) << i;
                        block_data[47] |= ((tmp_buffer2.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 48-55 abs quant from global memory.
                        block_data[48] |= ((tmp_buffer2.z >> 7) & 0x00000001) << i;
                        block_data[49] |= ((tmp_buffer2.z >> 6) & 0x00000001) << i;
                        block_data[50] |= ((tmp_buffer2.z >> 5) & 0x00000001) << i;
                        block_data[51] |= ((tmp_buffer2.z >> 4) & 0x00000001) << i;
                        block_data[52] |= ((tmp_buffer2.z >> 3) & 0x00000001) << i;
                        block_data[53] |= ((tmp_buffer2.z >> 2) & 0x00000001) << i;
                        block_data[54] |= ((tmp_buffer2.z >> 1) & 0x00000001) << i;
                        block_data[55] |= ((tmp_buffer2.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 56-63 abs quant from global memory.
                        block_data[56] |= ((tmp_buffer2.w >> 7) & 0x00000001) << i;
                        block_data[57] |= ((tmp_buffer2.w >> 6) & 0x00000001) << i;
                        block_data[58] |= ((tmp_buffer2.w >> 5) & 0x00000001) << i;
                        block_data[59] |= ((tmp_buffer2.w >> 4) & 0x00000001) << i;
                        block_data[60] |= ((tmp_buffer2.w >> 3) & 0x00000001) << i;
                        block_data[61] |= ((tmp_buffer2.w >> 2) & 0x00000001) << i;
                        block_data[62] |= ((tmp_buffer2.w >> 1) & 0x00000001) << i;
                        block_data[63] |= ((tmp_buffer2.w >> 0) & 0x00000001) << i;
                    }

                    // Restore quantized data.
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] * quant_bins[bin_choice];
                }

                // Write data back to global memory.
                data_idx_x = block_idx_x;
                for(uint i=0; i<8; i++)
                {
                    data_idx_y = block_idx_y * 8 + i;
                    for(uint k=0; k<8; k++)
                    {
                        data_idx_z = block_idx_z * 8 + k;
                        data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                        if(data_idx_y < dims.y && data_idx_z < dims.z) decData[data_idx] = block_data[i*8+k];
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}


__global__ void cuLSZ_compression_kernel_uint16_bsize64(const uint16_t* const __restrict__ oriData, 
                                                        unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ size_t excl_sum;
    __shared__ size_t base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    size_t thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.
    
    // Vector-quantization, Dynamic Binning Selection, Fixed-length Encoding.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }
            
            // Preparation for ratio profiling.
            uint zero_count = 0;
            uint zero_count_bins[4] = {0, 0, 0, 0};
            uint max_val1 = 0;
            uint max_val2 = 0;
            for(int i=0; i<64; i++)
            {
                uint val = block_data[i];
                zero_count += (val == 0);
                zero_count_bins[0] += (val < quant_bins[0]); // Base bin operation
                zero_count_bins[1] += (val < quant_bins[1]);
                zero_count_bins[2] += (val < quant_bins[2]);
                zero_count_bins[3] += (val < quant_bins[3]);
                max_val1 = (val > max_val1) ? val : max_val1;
                if(i%2)
                {
                    uint tmp_val = (block_data[i-1] + block_data[i]) / 2;
                    max_val2 = (tmp_val > max_val2) ? tmp_val : max_val2;
                }
            }

            // Compression algorithm selection and store meta data.
            float sparsity = (float)zero_count / 64;
            int pooling_choice = (sparsity > poolingTH);
            uint bin_choice = 0;
            // Progressively bin size selection.
            if(zero_count_bins[1]==zero_count_bins[0])
            {
                bin_choice = 1;
                if(zero_count_bins[2]==zero_count_bins[1])
                {
                    bin_choice = 2;
                    if(zero_count_bins[3]==zero_count_bins[2])
                    {
                        bin_choice = 3;
                    }
                }
            }

            // Store meta data.
            int max_quantized_val;
            int temp_rate = 0;
            if(pooling_choice)
            {
                max_quantized_val = max_val2 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 4;
                temp_rate = 0x80 | (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
            else
            {
                max_quantized_val = max_val1 / quant_bins[bin_choice];
                temp_rate = 32 - __clz((max_quantized_val));
                thread_ofs += temp_rate * 8;
                temp_rate = (bin_choice << 5) | temp_rate;
                fixed_rate[j] = (unsigned char)temp_rate;
                cmpBytes[block_idx] = fixed_rate[j];
            }
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            size_t loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also storing data to global memory.
    size_t base_cmp_byte_ofs = base_idx;
    size_t cmp_byte_ofs;
    size_t tmp_byte_ofs = 0;
    size_t cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Reading block data from memory, stored in block_data[64].
            uint block_data[64];
            data_idx_x = block_idx_x;
            for(uint i=0; i<8; i++) 
            {
                data_idx_y = block_idx_y * 8 + i;
                for(uint k=0; k<8; k++)
                {
                    data_idx_z = block_idx_z * 8 + k;
                    data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                    if(data_idx_y < dims.y && data_idx_z < dims.z)
                    {
                        block_data[i*8+k] = oriData[data_idx];
                    }
                    else
                    {
                        block_data[i*8+k] = 0;
                    }
                }
            }

            // Retrieve meta data.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;
            
            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            size_t prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                if(pooling_choice)
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) 
                    {
                        pooling_block_data[i] = (block_data[i*2] + block_data[i*2+1]) / 2;
                        pooling_block_data[i] = pooling_block_data[i] / quant_bins[bin_choice];
                    }

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer.x = 0;
                        tmp_buffer.y = 0;
                        tmp_buffer.z = 0;
                        tmp_buffer.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer.x = (((pooling_block_data[0] & mask) >> i) << 7) |
                                       (((pooling_block_data[1] & mask) >> i) << 6) |
                                       (((pooling_block_data[2] & mask) >> i) << 5) |
                                       (((pooling_block_data[3] & mask) >> i) << 4) |
                                       (((pooling_block_data[4] & mask) >> i) << 3) |
                                       (((pooling_block_data[5] & mask) >> i) << 2) |
                                       (((pooling_block_data[6] & mask) >> i) << 1) |
                                       (((pooling_block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer.y = (((pooling_block_data[8] & mask) >> i) << 7) |
                                       (((pooling_block_data[9] & mask) >> i) << 6) |
                                       (((pooling_block_data[10] & mask) >> i) << 5) |
                                       (((pooling_block_data[11] & mask) >> i) << 4) |
                                       (((pooling_block_data[12] & mask) >> i) << 3) |
                                       (((pooling_block_data[13] & mask) >> i) << 2) |
                                       (((pooling_block_data[14] & mask) >> i) << 1) |
                                       (((pooling_block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer.z = (((pooling_block_data[16] & mask) >> i) << 7) |
                                       (((pooling_block_data[17] & mask) >> i) << 6) |
                                       (((pooling_block_data[18] & mask) >> i) << 5) |
                                       (((pooling_block_data[19] & mask) >> i) << 4) |
                                       (((pooling_block_data[20] & mask) >> i) << 3) |
                                       (((pooling_block_data[21] & mask) >> i) << 2) |
                                       (((pooling_block_data[22] & mask) >> i) << 1) |
                                       (((pooling_block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer.w = (((pooling_block_data[24] & mask) >> i) << 7) |
                                       (((pooling_block_data[25] & mask) >> i) << 6) |
                                       (((pooling_block_data[26] & mask) >> i) << 5) |
                                       (((pooling_block_data[27] & mask) >> i) << 4) |
                                       (((pooling_block_data[28] & mask) >> i) << 3) |
                                       (((pooling_block_data[29] & mask) >> i) << 2) |
                                       (((pooling_block_data[30] & mask) >> i) << 1) |
                                       (((pooling_block_data[31] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer;
                        cmp_byte_ofs += 4;
                        mask <<= 1;  
                    }
                }
                else
                {
                    // Retrieve pooling data and quantize it.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] / quant_bins[bin_choice];

                    // Assign quant bit information for one block by bit-shuffle.
                    int mask = 1;
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Initialization.
                        tmp_buffer1.x = 0;
                        tmp_buffer1.y = 0;
                        tmp_buffer1.z = 0;
                        tmp_buffer1.w = 0;
                        tmp_buffer2.x = 0;
                        tmp_buffer2.y = 0;
                        tmp_buffer2.z = 0;
                        tmp_buffer2.w = 0;

                        // Get i-th bit in 0~7 data.
                        tmp_buffer1.x = (((block_data[0] & mask) >> i) << 7) |
                                        (((block_data[1] & mask) >> i) << 6) |
                                        (((block_data[2] & mask) >> i) << 5) |
                                        (((block_data[3] & mask) >> i) << 4) |
                                        (((block_data[4] & mask) >> i) << 3) |
                                        (((block_data[5] & mask) >> i) << 2) |
                                        (((block_data[6] & mask) >> i) << 1) |
                                        (((block_data[7] & mask) >> i) << 0);
                        
                        // Get i-th bit in 8~15 data.
                        tmp_buffer1.y = (((block_data[8] & mask) >> i) << 7) |
                                        (((block_data[9] & mask) >> i) << 6) |
                                        (((block_data[10] & mask) >> i) << 5) |
                                        (((block_data[11] & mask) >> i) << 4) |
                                        (((block_data[12] & mask) >> i) << 3) |
                                        (((block_data[13] & mask) >> i) << 2) |
                                        (((block_data[14] & mask) >> i) << 1) |
                                        (((block_data[15] & mask) >> i) << 0);

                        // Get i-th bit in 16~23 data.
                        tmp_buffer1.z = (((block_data[16] & mask) >> i) << 7) |
                                        (((block_data[17] & mask) >> i) << 6) |
                                        (((block_data[18] & mask) >> i) << 5) |
                                        (((block_data[19] & mask) >> i) << 4) |
                                        (((block_data[20] & mask) >> i) << 3) |
                                        (((block_data[21] & mask) >> i) << 2) |
                                        (((block_data[22] & mask) >> i) << 1) |
                                        (((block_data[23] & mask) >> i) << 0);

                        // Get i-th bit in 24~31 data.
                        tmp_buffer1.w = (((block_data[24] & mask) >> i) << 7) |
                                        (((block_data[25] & mask) >> i) << 6) |
                                        (((block_data[26] & mask) >> i) << 5) |
                                        (((block_data[27] & mask) >> i) << 4) |
                                        (((block_data[28] & mask) >> i) << 3) |
                                        (((block_data[29] & mask) >> i) << 2) |
                                        (((block_data[30] & mask) >> i) << 1) |
                                        (((block_data[31] & mask) >> i) << 0); 
                        
                        // Get i-th bit in 32~39 data.
                        tmp_buffer2.x = (((block_data[32] & mask) >> i) << 7) |
                                        (((block_data[33] & mask) >> i) << 6) |
                                        (((block_data[34] & mask) >> i) << 5) |
                                        (((block_data[35] & mask) >> i) << 4) |
                                        (((block_data[36] & mask) >> i) << 3) |
                                        (((block_data[37] & mask) >> i) << 2) |
                                        (((block_data[38] & mask) >> i) << 1) |
                                        (((block_data[39] & mask) >> i) << 0);
                        
                        // Get i-th bit in 40~47 data.
                        tmp_buffer2.y = (((block_data[40] & mask) >> i) << 7) |
                                        (((block_data[41] & mask) >> i) << 6) |
                                        (((block_data[42] & mask) >> i) << 5) |
                                        (((block_data[43] & mask) >> i) << 4) |
                                        (((block_data[44] & mask) >> i) << 3) |
                                        (((block_data[45] & mask) >> i) << 2) |
                                        (((block_data[46] & mask) >> i) << 1) |
                                        (((block_data[47] & mask) >> i) << 0);

                        // Get i-th bit in 48~55 data.
                        tmp_buffer2.z = (((block_data[48] & mask) >> i) << 7) |
                                        (((block_data[49] & mask) >> i) << 6) |
                                        (((block_data[50] & mask) >> i) << 5) |
                                        (((block_data[51] & mask) >> i) << 4) |
                                        (((block_data[52] & mask) >> i) << 3) |
                                        (((block_data[53] & mask) >> i) << 2) |
                                        (((block_data[54] & mask) >> i) << 1) |
                                        (((block_data[55] & mask) >> i) << 0);

                        // Get i-th bit in 56~63 data.
                        tmp_buffer2.w = (((block_data[56] & mask) >> i) << 7) |
                                        (((block_data[57] & mask) >> i) << 6) |
                                        (((block_data[58] & mask) >> i) << 5) |
                                        (((block_data[59] & mask) >> i) << 4) |
                                        (((block_data[60] & mask) >> i) << 3) |
                                        (((block_data[61] & mask) >> i) << 2) |
                                        (((block_data[62] & mask) >> i) << 1) |
                                        (((block_data[63] & mask) >> i) << 0);

                        // Move data to global memory via a vectorized manner.
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer1;
                        cmp_byte_ofs += 4;
                        reinterpret_cast<uchar4*>(cmpBytes)[cmp_byte_ofs/4] = tmp_buffer2;
                        cmp_byte_ofs += 4;
                        mask <<= 1; 
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}


__global__ void cuLSZ_decompression_kernel_uint16_bsize64(uint16_t* const __restrict__ decData, 
                                                        const unsigned char* const __restrict__ cmpBytes, 
                                                        volatile size_t* const __restrict__ cmpOffset, 
                                                        volatile size_t* const __restrict__ locOffset,
                                                        volatile int* const __restrict__ flag,
                                                        uint blockNum, const uint3 dims, 
                                                        const uint4 quantBins, const float poolingTH)
{
    __shared__ size_t excl_sum;
    __shared__ size_t base_idx;

    const uint tid = threadIdx.x;
    const uint bid = blockIdx.x;
    const uint idx = bid * blockDim.x + tid;
    const uint lane = idx & 0x1f;
    const uint warp = idx >> 5;
    const uint rate_ofs = (blockNum + 3) / 4 * 4;
    const uint dimyBlock = (dims.y + 7) / 8; // 8x8 blocks.
    const uint dimzBlock = (dims.z + 7) / 8; // 8x8 blocks, fastest dim.

    uint base_start_block_idx;
    uint block_idx;
    uint block_idx_x, block_idx_y, block_idx_z; // .z is the fastest dim.
    uint block_stride_per_slice;
    uint data_idx;
    uint data_idx_x, data_idx_y, data_idx_z;
    unsigned char fixed_rate[block_per_thread];
    uint quant_bins[4] = {quantBins.x, quantBins.y, quantBins.z, quantBins.w};
    size_t thread_ofs = 0;    // Derived from cuSZp, so use unsigned int instead of uint.

    // Obtain fixed-rate information for each block.
    base_start_block_idx = warp * 32 * block_per_thread;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;

        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Obtain block meta data.
            fixed_rate[j] = cmpBytes[block_idx];

            // Check if pooling.
            int pooling_choice = fixed_rate[j] >> 7;
            int temp_rate = fixed_rate[j] & 0x1f;
            if(pooling_choice) thread_ofs += temp_rate * 4;
            else thread_ofs += temp_rate * 8;
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            size_t loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also reading data from global memory.
    size_t base_cmp_byte_ofs = base_idx;
    size_t cmp_byte_ofs;
    size_t tmp_byte_ofs = 0;
    size_t cur_byte_ofs = 0;
    for(uint j=0; j<block_per_thread; j++)
    {
        // Block initialization.
        block_idx = base_start_block_idx + j * 32 + lane;
        block_stride_per_slice = dimyBlock * dimzBlock;
        block_idx_x = block_idx / block_stride_per_slice;
        block_idx_y = (block_idx % block_stride_per_slice) / dimzBlock;
        block_idx_z = (block_idx % block_stride_per_slice) % dimzBlock;
    
        // Avoid padding blocks.
        if(block_idx < blockNum)
        {
            // Initialization, guiding decoding process.
            int pooling_choice = fixed_rate[j] >> 7;
            uint bin_choice = (fixed_rate[j] & 0x60) >> 5;
            fixed_rate[j] &= 0x1f;

            // Restore index for j-th iteration.
            if(pooling_choice) tmp_byte_ofs = fixed_rate[j] * 4;
            else tmp_byte_ofs = fixed_rate[j] * 8;
            #pragma unroll 5
            for(int i=1; i<32; i<<=1)
            {
                int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
                if(lane >= i) tmp_byte_ofs += tmp;
            }
            size_t prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
            if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
            else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

            // Operation for each block, if zero block then do nothing.
            if(fixed_rate[j])
            {
                // Buffering decompressed block data.
                uint block_data[64];

                // Read data and shuffle it back from global memory.
                if(pooling_choice)
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer;
                    uint pooling_block_data[32];
                    for(int i=0; i<32; i++) pooling_block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        pooling_block_data[0] |= ((tmp_buffer.x >> 7) & 0x00000001) << i;
                        pooling_block_data[1] |= ((tmp_buffer.x >> 6) & 0x00000001) << i;
                        pooling_block_data[2] |= ((tmp_buffer.x >> 5) & 0x00000001) << i;
                        pooling_block_data[3] |= ((tmp_buffer.x >> 4) & 0x00000001) << i;
                        pooling_block_data[4] |= ((tmp_buffer.x >> 3) & 0x00000001) << i;
                        pooling_block_data[5] |= ((tmp_buffer.x >> 2) & 0x00000001) << i;
                        pooling_block_data[6] |= ((tmp_buffer.x >> 1) & 0x00000001) << i;
                        pooling_block_data[7] |= ((tmp_buffer.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        pooling_block_data[8] |= ((tmp_buffer.y >> 7) & 0x00000001) << i;
                        pooling_block_data[9] |= ((tmp_buffer.y >> 6) & 0x00000001) << i;
                        pooling_block_data[10] |= ((tmp_buffer.y >> 5) & 0x00000001) << i;
                        pooling_block_data[11] |= ((tmp_buffer.y >> 4) & 0x00000001) << i;
                        pooling_block_data[12] |= ((tmp_buffer.y >> 3) & 0x00000001) << i;
                        pooling_block_data[13] |= ((tmp_buffer.y >> 2) & 0x00000001) << i;
                        pooling_block_data[14] |= ((tmp_buffer.y >> 1) & 0x00000001) << i;
                        pooling_block_data[15] |= ((tmp_buffer.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        pooling_block_data[16] |= ((tmp_buffer.z >> 7) & 0x00000001) << i;
                        pooling_block_data[17] |= ((tmp_buffer.z >> 6) & 0x00000001) << i;
                        pooling_block_data[18] |= ((tmp_buffer.z >> 5) & 0x00000001) << i;
                        pooling_block_data[19] |= ((tmp_buffer.z >> 4) & 0x00000001) << i;
                        pooling_block_data[20] |= ((tmp_buffer.z >> 3) & 0x00000001) << i;
                        pooling_block_data[21] |= ((tmp_buffer.z >> 2) & 0x00000001) << i;
                        pooling_block_data[22] |= ((tmp_buffer.z >> 1) & 0x00000001) << i;
                        pooling_block_data[23] |= ((tmp_buffer.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        pooling_block_data[24] |= ((tmp_buffer.w >> 7) & 0x00000001) << i;
                        pooling_block_data[25] |= ((tmp_buffer.w >> 6) & 0x00000001) << i;
                        pooling_block_data[26] |= ((tmp_buffer.w >> 5) & 0x00000001) << i;
                        pooling_block_data[27] |= ((tmp_buffer.w >> 4) & 0x00000001) << i;
                        pooling_block_data[28] |= ((tmp_buffer.w >> 3) & 0x00000001) << i;
                        pooling_block_data[29] |= ((tmp_buffer.w >> 2) & 0x00000001) << i;
                        pooling_block_data[30] |= ((tmp_buffer.w >> 1) & 0x00000001) << i;
                        pooling_block_data[31] |= ((tmp_buffer.w >> 0) & 0x00000001) << i;
                    }

                    // Assign data back to block data.
                    for(int i=0; i<32; i++)
                    {
                        block_data[i*2] = pooling_block_data[i] * quant_bins[bin_choice];
                        block_data[i*2+1] = block_data[i*2];
                    }
                }
                else
                {
                    // Initialize buffer.
                    uchar4 tmp_buffer1, tmp_buffer2;
                    for(int i=0; i<64; i++) block_data[i] = 0;

                    // Shuffle data back.
                    for(int i=0; i<fixed_rate[j]; i++)
                    {
                        // Read data from global memory.
                        tmp_buffer1 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;
                        tmp_buffer2 = reinterpret_cast<const uchar4*>(cmpBytes)[cmp_byte_ofs/4];
                        cmp_byte_ofs += 4;

                        // Get ith bit in 0~7 abs quant from global memory.
                        block_data[0] |= ((tmp_buffer1.x >> 7) & 0x00000001) << i;
                        block_data[1] |= ((tmp_buffer1.x >> 6) & 0x00000001) << i;
                        block_data[2] |= ((tmp_buffer1.x >> 5) & 0x00000001) << i;
                        block_data[3] |= ((tmp_buffer1.x >> 4) & 0x00000001) << i;
                        block_data[4] |= ((tmp_buffer1.x >> 3) & 0x00000001) << i;
                        block_data[5] |= ((tmp_buffer1.x >> 2) & 0x00000001) << i;
                        block_data[6] |= ((tmp_buffer1.x >> 1) & 0x00000001) << i;
                        block_data[7] |= ((tmp_buffer1.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 8~15 abs quant from global memory.
                        block_data[8] |= ((tmp_buffer1.y >> 7) & 0x00000001) << i;
                        block_data[9] |= ((tmp_buffer1.y >> 6) & 0x00000001) << i;
                        block_data[10] |= ((tmp_buffer1.y >> 5) & 0x00000001) << i;
                        block_data[11] |= ((tmp_buffer1.y >> 4) & 0x00000001) << i;
                        block_data[12] |= ((tmp_buffer1.y >> 3) & 0x00000001) << i;
                        block_data[13] |= ((tmp_buffer1.y >> 2) & 0x00000001) << i;
                        block_data[14] |= ((tmp_buffer1.y >> 1) & 0x00000001) << i;
                        block_data[15] |= ((tmp_buffer1.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 16-23 abs quant from global memory.
                        block_data[16] |= ((tmp_buffer1.z >> 7) & 0x00000001) << i;
                        block_data[17] |= ((tmp_buffer1.z >> 6) & 0x00000001) << i;
                        block_data[18] |= ((tmp_buffer1.z >> 5) & 0x00000001) << i;
                        block_data[19] |= ((tmp_buffer1.z >> 4) & 0x00000001) << i;
                        block_data[20] |= ((tmp_buffer1.z >> 3) & 0x00000001) << i;
                        block_data[21] |= ((tmp_buffer1.z >> 2) & 0x00000001) << i;
                        block_data[22] |= ((tmp_buffer1.z >> 1) & 0x00000001) << i;
                        block_data[23] |= ((tmp_buffer1.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 24-31 abs quant from global memory.
                        block_data[24] |= ((tmp_buffer1.w >> 7) & 0x00000001) << i;
                        block_data[25] |= ((tmp_buffer1.w >> 6) & 0x00000001) << i;
                        block_data[26] |= ((tmp_buffer1.w >> 5) & 0x00000001) << i;
                        block_data[27] |= ((tmp_buffer1.w >> 4) & 0x00000001) << i;
                        block_data[28] |= ((tmp_buffer1.w >> 3) & 0x00000001) << i;
                        block_data[29] |= ((tmp_buffer1.w >> 2) & 0x00000001) << i;
                        block_data[30] |= ((tmp_buffer1.w >> 1) & 0x00000001) << i;
                        block_data[31] |= ((tmp_buffer1.w >> 0) & 0x00000001) << i;

                        // Get ith bit in 32~39 abs quant from global memory.
                        block_data[32] |= ((tmp_buffer2.x >> 7) & 0x00000001) << i;
                        block_data[33] |= ((tmp_buffer2.x >> 6) & 0x00000001) << i;
                        block_data[34] |= ((tmp_buffer2.x >> 5) & 0x00000001) << i;
                        block_data[35] |= ((tmp_buffer2.x >> 4) & 0x00000001) << i;
                        block_data[36] |= ((tmp_buffer2.x >> 3) & 0x00000001) << i;
                        block_data[37] |= ((tmp_buffer2.x >> 2) & 0x00000001) << i;
                        block_data[38] |= ((tmp_buffer2.x >> 1) & 0x00000001) << i;
                        block_data[39] |= ((tmp_buffer2.x >> 0) & 0x00000001) << i;

                        // Get ith bit in 40~47 abs quant from global memory.
                        block_data[40] |= ((tmp_buffer2.y >> 7) & 0x00000001) << i;
                        block_data[41] |= ((tmp_buffer2.y >> 6) & 0x00000001) << i;
                        block_data[42] |= ((tmp_buffer2.y >> 5) & 0x00000001) << i;
                        block_data[43] |= ((tmp_buffer2.y >> 4) & 0x00000001) << i;
                        block_data[44] |= ((tmp_buffer2.y >> 3) & 0x00000001) << i;
                        block_data[45] |= ((tmp_buffer2.y >> 2) & 0x00000001) << i;
                        block_data[46] |= ((tmp_buffer2.y >> 1) & 0x00000001) << i;
                        block_data[47] |= ((tmp_buffer2.y >> 0) & 0x00000001) << i;

                        // Get ith bit in 48-55 abs quant from global memory.
                        block_data[48] |= ((tmp_buffer2.z >> 7) & 0x00000001) << i;
                        block_data[49] |= ((tmp_buffer2.z >> 6) & 0x00000001) << i;
                        block_data[50] |= ((tmp_buffer2.z >> 5) & 0x00000001) << i;
                        block_data[51] |= ((tmp_buffer2.z >> 4) & 0x00000001) << i;
                        block_data[52] |= ((tmp_buffer2.z >> 3) & 0x00000001) << i;
                        block_data[53] |= ((tmp_buffer2.z >> 2) & 0x00000001) << i;
                        block_data[54] |= ((tmp_buffer2.z >> 1) & 0x00000001) << i;
                        block_data[55] |= ((tmp_buffer2.z >> 0) & 0x00000001) << i;

                        // Get ith bit in 56-63 abs quant from global memory.
                        block_data[56] |= ((tmp_buffer2.w >> 7) & 0x00000001) << i;
                        block_data[57] |= ((tmp_buffer2.w >> 6) & 0x00000001) << i;
                        block_data[58] |= ((tmp_buffer2.w >> 5) & 0x00000001) << i;
                        block_data[59] |= ((tmp_buffer2.w >> 4) & 0x00000001) << i;
                        block_data[60] |= ((tmp_buffer2.w >> 3) & 0x00000001) << i;
                        block_data[61] |= ((tmp_buffer2.w >> 2) & 0x00000001) << i;
                        block_data[62] |= ((tmp_buffer2.w >> 1) & 0x00000001) << i;
                        block_data[63] |= ((tmp_buffer2.w >> 0) & 0x00000001) << i;
                    }

                    // Restore quantized data.
                    for(int i=0; i<64; i++) block_data[i] = block_data[i] * quant_bins[bin_choice];
                }

                // Write data back to global memory.
                data_idx_x = block_idx_x;
                for(uint i=0; i<8; i++)
                {
                    data_idx_y = block_idx_y * 8 + i;
                    for(uint k=0; k<8; k++)
                    {
                        data_idx_z = block_idx_z * 8 + k;
                        data_idx = data_idx_x * dims.y * dims.z + data_idx_y * dims.z + data_idx_z;
                        if(data_idx_y < dims.y && data_idx_z < dims.z) decData[data_idx] = block_data[i*8+k];
                    }
                }
            }

            // Index updating across different iterations.
            cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
        }
    }
}