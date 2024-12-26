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
    int error = 0;
    int status=0;

    // Check if enough arguments are provided
    if (argc < 9) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "   %s -i oriFilePath -d dims.x dims.y dims.z -e error -o decFilePath\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "   -i oriFilePath: Path to the original data file\n");
        fprintf(stderr, "   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.\n");
        fprintf(stderr, "   -e errors: An integer number of tolerant error\n");
        fprintf(stderr, "   -o decFilePath: Path to the decompressed data file (optional).\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "   %s -i data/cssi.bin -d 600 1813 1558 -e 5\n", argv[0]);
        fprintf(stderr, "   %s -i data/cssi.bin -d 600 1813 1558 -e 3 -o data/cssi-dec.bin\n", argv[0]);
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
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            error = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strncpy(decFilePath, argv[++i], sizeof(decFilePath) - 1);
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }
    }

    // Verify mandatory arguments
    if (strlen(oriFilePath) == 0 || dims.x == 0 || dims.y == 0 || dims.z == 0 || error == 0) {
        fprintf(stderr, "Error: Missing mandatory arguments.\n");
        return EXIT_FAILURE;
    }

    // Yafan is testing parsed arguments for confirmation
    printf("Original File Path: %s\n", oriFilePath);
    printf("Dimensions: %u x %u x %u\n", dims.x, dims.y, dims.z);
    printf("Error Tolerance: %d\n", error);
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
    float* dup_oriData = NULL;
    float* dup_decData = NULL;
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
    dup_oriData = (float*)malloc(nbEle * sizeof(float));
    dup_decData = (float*)malloc(nbEle * sizeof(float));
    // Converting uint32_t to float
    for (size_t i = 0; i < nbEle; i++) {
        dup_oriData[i] = (float)oriData[i];
    }
    
    // Input data preparation on GPU.
    float* d_oriData;
    float* d_decData;
    unsigned char* d_cmpBytes;
    cudaMalloc((void**)&d_oriData, nbEle*sizeof(float));
    cudaMemcpy(d_oriData, dup_oriData, nbEle*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, nbEle*sizeof(float));
    cudaMalloc((void**)&d_cmpBytes, nbEle*sizeof(float));

    // Initialize CUDA stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup for NVIDIA GPU.
    for(int i=0; i<3; i++) cuSZp_compress_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, (float)error, stream);
    printf("GPU warmup finished!\n\n");

    // cuSZp compression.
    timer_GPU.StartCounter();
    cuSZp_compress_plain_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, (float)error, stream);
    float cmpTime = timer_GPU.GetCounter();

    // Transfer compressed data to CPU then back to GPU, making sure compression ratio is correct.
    // No need to add this part for real-world usages, this is only for testing compresion ratio correcness.
    unsigned char* cmpBytes_dup = (unsigned char*)malloc(cmpSize*sizeof(unsigned char));
    cudaMemcpy(cmpBytes_dup, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemset(d_cmpBytes, 0, nbEle*sizeof(uint32_t)); // set to zero for double check.
    cudaMemcpy(d_cmpBytes, cmpBytes_dup, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice); // copy back to GPU.

    // cuSZp decompression.
    timer_GPU.StartCounter();
    cuSZp_decompress_plain_f32(d_decData, d_cmpBytes, nbEle, cmpSize, (float)error, stream);
    float decTime = timer_GPU.GetCounter();

    // Print results.
    printf("Dataset information:\n");
    printf("  - dims:   %u x %u x %u\n", dims.x, dims.y, dims.z);
    printf("  - length: %zu\n", nbEle);
    printf("  - size:   %f GB\n", nbEle*sizeof(uint32_t)/1024.0/1024.0/1024.0);
    printf("Input arguments:\n");
    printf("  - error:  %d\n", error);
    printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(uint32_t)/1024.0/1024.0)/cmpTime);
    printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(uint32_t)/1024.0/1024.0)/decTime);
    printf("cuSZp compression ratio: %f\n", (nbEle*sizeof(uint32_t)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));

    // Write reconstructed data if needed.
    if(strlen(decFilePath) > 0) {
        // Converting data back.
        cudaMemcpy(dup_decData, d_decData, nbEle*sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nbEle; i++) {
            decData[i] = (uint32_t)dup_decData[i];
        }
        writeUIntData_inBytes_Yafan(decData, nbEle, decFilePath, &status);
    }

    free(oriData);
    free(decData);
    free(cmpBytes);
    free(cmpBytes_dup);
    free(dup_oriData);
    free(dup_decData);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);
    return 0;
}