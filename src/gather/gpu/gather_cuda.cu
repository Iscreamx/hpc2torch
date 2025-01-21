#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void gather_kernel(
    const T* data,
    const size_t* indices, 
    T* output,           
    int n_ind,
    int n_output,
    int dim,
    int pass,
    int num_per_pass) {        
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_ind) return;

    int data_start = 0;
    int output_start = 0;

    for(int i = 0; i < pass; i++) {
        data_start = i * num_per_pass * dim + indices[tid] * num_per_pass;
        output_start = tid * num_per_pass + n_output / pass * i;

        for (int j = 0; j < num_per_pass; ++j) {
            output[output_start + j] = data[data_start + j];
        }
    }
}

template <typename T>
void launch_gather(
    const T* data,
    const size_t* indices,
    T* output,
    int pass,
    int num_per_pass,
    int dim,
    int n_ind,
    int n_output) {

    int block_size = 256;
    int grid_size = (n_ind + block_size - 1) / block_size;

    gather_kernel<<<grid_size, block_size>>>(
        data, indices, output, n_ind,n_output, dim, pass, num_per_pass);

}

extern "C" {
    void launch_gather_f32(
        const void* data,
        const size_t* indices,
        void* output,
        int h_pass,
        int h_num_per_pass,
        int dim,
        int num_ind,
        int num) {
        launch_gather((float *)data, indices, (float *)output, h_pass,
            h_num_per_pass, dim, num_ind, num);
    }

    void launch_gather_f16(
        const void* data,
        const size_t* indices,
        const void* output,
        int h_pass,
        int h_num_per_pass,
        int dim,
        int num_ind,
        int num) {
        launch_gather((half *)data, indices, (half *)output, h_pass,
            h_num_per_pass, dim, num_ind, num);
    }
}
/*
ncu --kernel-name "gather_kernel" python /home/iscreamx/project/hpc2torch/test/gather.py
*/

