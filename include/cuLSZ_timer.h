#ifndef CULSZ_INCLUDE_CULSZ_TIMER_H
#define CULSZ_INCLUDE_CULSZ_TIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
    cudaEvent_t start;
    cudaEvent_t stop;
};

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:
        TimingGPU();
        ~TimingGPU();
        void StartCounter();
        void StartCounterFlags();
        float GetCounter();

};

#endif // CULSZ_INCLUDE_CULSZ_TIMER_H