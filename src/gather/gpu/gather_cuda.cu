#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief CUDA 核函数，用于从输入数据中根据索引收集数据到输出数组中。
 *
 * @tparam T 数据类型
 * @param data 输入数据数组的指针
 * @param indices 索引数组的指针，指定从 `data` 中收集数据的位置
 * @param output 输出数据数组的指针，用于存储收集的结果
 * @param n_ind 索引数组的元素数量
 * @param n_output 输出数组的元素数量
 * @param dim 输入数据 `data` 在axis维度上的大小(data_dims[axis])
 * @param pass 当前处理的批次（用于分批次处理数据）
 * @param num_per_pass 每个批次处理的元素数量
 */
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


    /*
    对于indices数组的每个索引,考虑其从data数组中收集数据的情况
    若data的维度为(a, b, c, d), indices的维度为(e, f, g)
        若axis为1, 则每个索引收集的数据个数为a x c x d, pass = a, num_per_pass = c x d, output_dims = a x e x f x g x c x d
        若axis为2, 则每个索引收集的数据个数为a x b x d, pass = a x b, num_per_pass = d, output_dims = a x b x e x f x g x d
        若axis为3, 则每个索引收集的数据个数为a x b x c, pass = a x b x c, num_per_pass = 1, output_dims = a x b x c x e x f x g
    对于indices中每个索引的每一个pass, 从data数组中收集num_per_pass个数据, 这些数据地址连续, 可以用多个线程并行处理
    indices中的不同索引, 可能取到data中重复的数据, 但每次都需要重新从全局内存中加载, 可能有大量的冗余读取
    另一种可能的实现方式是, 对于每num_per_pass个数据, 遍历索引数组, 将相同的索引的数据收集到output数组中, 但如果indices中的索引数量很多, 
    甚至超过data的大小, 也需要多次访存

    原始实现:
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
