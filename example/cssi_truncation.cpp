#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuLSZ_entry.h>
#include <cuLSZ_timer.h>
#include <cuLSZ_utility.h>

int main(int argc, char* argv[])
{
    // Read input information.
    char oriFilePath[640] = {0};
    char decFilePath[640] = {0};
    uint3 dims = {0, 0, 0};
    int threshold = 0;
    int status=0;

    // Check if enough arguments are provided
    if (argc < 9) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "   %s -i oriFilePath -d dims.x dims.y dims.z -t threshold -o decFilePath\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "   -i oriFilePath: Path to the original data file\n");
        fprintf(stderr, "   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.\n");
        fprintf(stderr, "   -t threshold: An integer number, below this threshold data will be set as 0.\n");
        fprintf(stderr, "   -o decFilePath: Path to the decompressed data file (optional).\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "   %s -i data/cssi.bin -d 600 1813 1558 -t 5\n", argv[0]);
        fprintf(stderr, "   %s -i data/cssi.bin -d 600 1813 1558 -t 3 -o data/cssi-dec.bin\n", argv[0]);
        return 1;
    }

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            strncpy(oriFilePath, argv[++i], sizeof(oriFilePath) - 1);
        } else if (strcmp(argv[i], "-d") == 0 && i + 3 < argc) {
            dims.x = atoi(argv[++i]);
            dims.y = atoi(argv[++i]);
            dims.z = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strncpy(decFilePath, argv[++i], sizeof(decFilePath) - 1);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }
    }

    // Verify mandatory arguments
    if (strlen(oriFilePath) == 0 || dims.x == 0 || dims.y == 0 || dims.z == 0 || threshold == 0) {
        fprintf(stderr, "Error: Missing mandatory arguments.\n");
        return EXIT_FAILURE;
    }

    // Yafan is testing parsed arguments for confirmation
    printf("Original File Path: %s\n", oriFilePath);
    printf("Dimensions: %u x %u x %u\n", dims.x, dims.y, dims.z);
    printf("Threshold: %d\n", threshold);
    if (strlen(decFilePath) > 0) {
        printf("Decompressed File Path: %s\n", decFilePath);
    } else {
        printf("Decompressed File Path: Not provided (optional).\n");
    }

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    uint32_t* oriData = NULL;
    uint32_t* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    oriData = readUInt32Data_Yafan(oriFilePath, &nbEle, &status);
    if(nbEle != dims.x * dims.y * dims.z) {
        fprintf(stderr, "Error: The number of elements in the original data does not match the dimensions\n");
        return 1;
    }
    decData = (uint32_t*)malloc(nbEle * sizeof(uint32_t));
    cmpBytes = (unsigned char*)malloc(nbEle * sizeof(uint32_t));

    // Input data preparation on GPU.
    uint32_t* d_oriData;
    uint32_t* d_decData;
    unsigned char* d_cmpBytes;
    cudaMalloc((void**)&d_oriData, nbEle*sizeof(uint32_t));
    cudaMemcpy(d_oriData, oriData, nbEle*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, nbEle*sizeof(uint32_t));
    cudaMalloc((void**)&d_cmpBytes, nbEle*sizeof(uint32_t));

    // Initialize CUDA stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup for NVIDIA GPU.
    for(int i=0; i<3; i++) cuSZp_compress_plain_f32_truncation_cssi(d_oriData, d_cmpBytes, nbEle, &cmpSize, (uint)threshold, stream);
    printf("GPU warmup finished!\n\n");

    // cuSZp compression.
    timer_GPU.StartCounter();
    cuSZp_compress_plain_f32_truncation_cssi(d_oriData, d_cmpBytes, nbEle, &cmpSize, (uint)threshold, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, nbEle*sizeof(uint32_t)); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice); // copy back to GPU.

    // cuSZp decompression.
    timer_GPU.StartCounter();
    cuSZp_decompress_plain_f32_truncation_cssi(d_decData, d_cmpBytes, nbEle, cmpSize, (uint)threshold, stream);
    float decTime = timer_GPU.GetCounter();

    // Print results.
    printf("Dataset information:\n");
    printf("  - dims:   %u x %u x %u\n", dims.x, dims.y, dims.z);
    printf("  - length: %zu\n", nbEle);
    printf("  - size:   %f GB\n", nbEle*sizeof(uint32_t)/1024.0/1024.0/1024.0);
    printf("Input arguments:\n");
    printf("  - threshold:  %d\n", threshold);
    printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(uint32_t)/1024.0/1024.0)/cmpTime);
    printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(uint32_t)/1024.0/1024.0)/decTime);
    printf("cuSZp compression ratio: %f\n", (nbEle*sizeof(uint32_t)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));

    // Write reconstructed data if needed.
    if(strlen(decFilePath) > 0) {
        // Converting data back.
        cudaMemcpy(decData, d_decData, nbEle*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        writeUIntData_inBytes_Yafan(decData, nbEle, decFilePath, &status);
    }

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);
    return 0;
}