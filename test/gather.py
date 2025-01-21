import torch
import ctypes
import numpy as np
import argparse
import performance
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor

def calculate_output_dims_and_num_elements(inputShape, indexShape, axis):
    """
    计算输出张量的形状和元素个数
    """
    output_dims = list(inputShape)
    output_dims[axis] = indexShape[0]
    if len(indexShape) > 1:
        output_dims[axis:axis + 1] = list(indexShape)
    num_elements = 1
    for dim in output_dims:
        num_elements *= dim
    return output_dims, num_elements

def calculate_h_pass_and_h_num_per_pass(inputShape, axis):
    """
    计算 h_pass 和 h_num_per_pass
    """
    h_pass = 1
    for i in range(axis):
        h_pass *= inputShape[i]
    
    h_num_per_pass = 1
    for i in range(axis + 1, len(inputShape)):
        h_num_per_pass *= inputShape[i]
    
    return h_pass, h_num_per_pass

def test(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing gather on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)
    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)

    data_rank = len(inputShape)
    index_rank = len(indexShape)

    output_dims, num_elements = calculate_output_dims_and_num_elements(inputShape, indexShape, axis)
    # print("Output shape:", output_dims)

    outTensor = gather(data_rank, axis, inputTensor, indexTensor)

    Q_output = torch.zeros(outTensor.shape, device=device, dtype=test_dtype)

    num_ind = 1
    for dim in indexTensor.shape:
        num_ind *= dim

    dim = inputTensor.shape[axis]

    h_pass, h_num_per_pass = calculate_h_pass_and_h_num_per_pass(inputShape, axis)

    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_size_t))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    if test_dtype == torch.float32:
        if device == "cuda":
            torch_gather_time = performance.CudaProfile((gather, (data_rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((lib.launch_gather_f32, (input_ptr, index_ptr, output_ptr, h_pass, h_num_per_pass, dim, num_ind, num_elements)))
    if test_dtype == torch.float16:
        if device == "cuda":
            torch_gather_time = performance.CudaProfile((gather, (data_rank, axis, inputTensor, indexTensor)))
            custom_gather_time = performance.CudaProfile((lib.launch_gather_f16, (input_ptr, index_ptr, output_ptr, h_pass, h_num_per_pass, dim, num_ind, num_elements)))

    performance.logBenchmark(torch_gather_time, custom_gather_time)

    # Verification
    tmpa = outTensor.to('cpu').numpy().flatten()
    tmpb = Q_output.to('cpu').numpy().flatten()
    atol = max(abs(tmpa - tmpb))
    rtol = atol / max(abs(tmpb) + 1e-8)
    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

parser = argparse.ArgumentParser(description="Test gather on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        #((3, 2), (2, 2), 0, torch.float32, "cuda"),
        #((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),

        #((3, 2), (2, 2), 0, torch.float16, "cuda"),
        #((3, 2), (1, 2), 1, torch.float16, "cuda"),
        #((50257, 768), (16, 1024), 0, torch.float16, "cuda"),

        #((9, 9,9,9), (2,7,2), 0, torch.float32, "cuda"),
        #((9, 9,9,9), (2,7,2), 1, torch.float32, "cuda"),
        #((9, 9,9,9), (2,7,2), 0, torch.float16, "cuda"),
        #((9, 9,9,9), (2,7,2), 1, torch.float16, "cuda"),

        #((6, 4, 5, 6), (2, 3, 2), 3, torch.float32, "cuda"),
        #((4, 5, 4, 4, 5), (2, 3, 2), 4, torch.float32, "cuda"),

]
filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)