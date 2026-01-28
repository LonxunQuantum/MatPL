#include <torch/extension.h>
#include "../include/calculate_compress_grad.h"

void torch_launch_calculate_compress_grad(
                        const torch::Tensor &f2,
                        const torch::Tensor &coefficient,
                        const torch::Tensor &grad_output,
                        int64_t sij_num,
                        int64_t layer_node,
                        int64_t coe_num,
                        const torch::Tensor &Grad
)
{
    auto dtype = f2.dtype();
    assert(coefficient.dtype() == dtype);
    // assert(G.dtype() == dtype);
    if (dtype == torch::kFloat32)
    {
        launch_calculate_compress_grad<float>(
            (const float *) f2.data_ptr(),
            (const float *) coefficient.data_ptr(),
            (const float *) grad_output.data_ptr(),
            sij_num, layer_node, coe_num,
            (float *) Grad.data_ptr()
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_compress_grad<double>(
            (const double *) f2.data_ptr(),
            (const double *) coefficient.data_ptr(),
            (const double *) grad_output.data_ptr(),
            sij_num, layer_node, coe_num,
            (double *) Grad.data_ptr()
        );
    }
    else
        printf("data type error!");
}
