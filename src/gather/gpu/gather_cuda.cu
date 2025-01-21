#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void gather_kernel(
    T* data,
    const size_t* indices, 
    T* output,           
    int n_ind,
    int n_output,
    int dim,
    int pass,
    int num_per_pass) 
{
    int tid = blockIdx.x + blockIdx.y * gridDim.x;
    if (tid >= n_ind) return;
    int data_start, output_start;

    // 原始实现
    /*
    for(int i = 0; i < pass; i++) {
        data_start = i * num_per_pass * dim + indices[tid] * num_per_pass;
        output_start = tid * num_per_pass + n_output / pass * i;

        for (int j = 0; j < num_per_pass; ++j) {
            output[output_start + j] = data[data_start + j];
        }
    }
    */

    for (int i = threadIdx.y; i < pass; i += blockDim.y) {
        data_start = i * num_per_pass * dim + indices[tid] * num_per_pass;
        output_start = tid * num_per_pass + (n_output / pass) * i;

        bool isFloat = std::is_same<T, float>::value;
        bool isHalf  = std::is_same<T, half>::value;

        size_t dataPtr   = reinterpret_cast<size_t>(data + data_start);
        size_t outputPtr = reinterpret_cast<size_t>(output + output_start);

        bool isAlignedFloat4 = (isFloat && (dataPtr % 16 == 0) && (outputPtr % 16 == 0));
        bool isAlignedHalf2  = (isHalf && (dataPtr % 4  == 0) && (outputPtr % 4  == 0));

        if (isAlignedFloat4) {
            int stride = blockDim.x * 4;
            for (int j = threadIdx.x * 4; j < num_per_pass; j += stride) {
                int remain = num_per_pass - j;
                if (remain >= 4) {
                    reinterpret_cast<float4*>(output + output_start)[j >> 2] =
                        reinterpret_cast<const float4*>(data + data_start)[j >> 2];
                } else {
                    for (int k = 0; k < remain; k++) {
                        output[output_start + j + k] = data[data_start + j + k];
                    }
                }
            }
        } else if (isAlignedHalf2) {
            int stride = blockDim.x * 2;
            for (int j = threadIdx.x * 2; j < num_per_pass; j += stride) {
                int remain = num_per_pass - j;
                if (remain >= 2) {
                    reinterpret_cast<half2*>(output + output_start)[j >> 1] =
                        reinterpret_cast<const half2*>(data + data_start)[j >> 1];
                } else {
                    for (int k = 0; k < remain; k++) {
                        output[output_start + j + k] = data[data_start + j + k];
                    }
                }
            }
        } else {
            for (int j = threadIdx.x; j < num_per_pass; j += blockDim.x) {
                output[output_start + j] = data[data_start + j];
            }
        }
    }
}

template <typename T>
void launch_gather(
    T* data,
    const size_t* indices,
    T* output,
    int pass,
    int num_per_pass,
    int dim,
    int n_ind,
    int n_output) {

    dim3 blockDim (128, 1, 1);
    dim3 gridDim (n_ind, 1, 1);

    gather_kernel<<<gridDim, blockDim>>>(
        data, indices, output, n_ind, n_output, dim, pass, num_per_pass);
}

extern "C" {
    void launch_gather_f32(
        void* data,
        const size_t* indices,
        void* output,
        int pass,
        int num_per_pass,
        int dim,
        int num_ind,
        int num) {
        launch_gather((float *)data, indices, (float *)output, pass,
            num_per_pass, dim, num_ind, num);
    }

    void launch_gather_f16(
        void* data,
        const size_t* indices,
        const void* output,
        int pass,
        int num_per_pass,
        int dim,
        int num_ind,
        int num) {
        launch_gather((half *)data, indices, (half *)output, pass,
            num_per_pass, dim, num_ind, num);
    }
}
