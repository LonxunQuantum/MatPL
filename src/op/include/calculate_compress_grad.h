template<typename DType>
void launch_calculate_compress_grad(
    const DType * f2,
    const DType * coefficient,
    const DType * grad_output,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    DType * Grad
);

