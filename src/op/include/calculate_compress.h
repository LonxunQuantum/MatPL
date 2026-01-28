template<typename DType>
void launch_calculate_compress(
    const DType * f2,
    const DType * coefficient,
    const int sij_num,
    const int layer_node,
    const int coe_num,
    DType * G,
    const int device
);
