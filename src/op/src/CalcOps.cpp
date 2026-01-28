#include <torch/torch.h>
#include <iostream>
#include "../include/CalcOps.h"
#include "../include/op_declare.h"

torch::autograd::variable_list CalculateForceFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        int nghost = nghost_tensor.item<int>();
        auto force = torch::zeros({batch_size, natoms + nghost, 3}, dE.options());
        force.slice(1, 0, natoms) = F;
        torch_launch_calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, force, nghost);
        return {force};
        
    }

torch::autograd::variable_list CalculateForceFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_force_grad(list_neigh, Ri_d, grad_output[0], batch_size, natoms, neigh_num, grad);
        
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateForce::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        ctx->save_for_backward({list_neigh, dE, Ri_d});
        return CalculateForceFuncs::forward(list_neigh, dE, Ri_d, F, nghost_tensor);
    }

torch::autograd::variable_list CalculateForce::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Ri_d = saved[2];
        return CalculateForceFuncs::backward(grad_output, list_neigh, dE, Ri_d);
    }
    
torch::autograd::variable_list calculateForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F,
    at::Tensor nghost_tensor) 
    {
        return CalculateForce::apply(list_neigh, dE, Ri_d, F, nghost_tensor);
    }

// the nep force
torch::autograd::variable_list CalculateNepForce::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F) 
    {
        auto dims = list_neigh.sizes();
        int natoms = dims[0];
        int neigh_num = dims[1];
        torch_launch_calculate_nepforce(
            list_neigh, 
            dE, 
            Ri_d, 
            natoms, 
            neigh_num, 
            F);
        ctx->save_for_backward({list_neigh, dE, Ri_d});
        return {F};
    }

torch::autograd::variable_list CalculateNepForce::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output)
    {
        // std::cout << "CalculateNepForce::backward in grad_output shape: " << grad_output[0].sizes() << std::endl;
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Ri_d = saved[2];

        auto dims = list_neigh.sizes();
        int natoms = dims[0];
        int neigh_num = dims[1];
        auto grad = torch::zeros({natoms, neigh_num, 4}, grad_output[0].options());
        // std::cout << "CalculateNepForce::backward grad_output shape: " << grad_output[0].sizes() << std::endl;

        // auto gradin = grad_output[0].to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     printf("backforce grad_output[%d][:] = %f %f %f\n", i, 
        //     gradin_data[i * gradin_dim1 + 0],
        //     gradin_data[i * gradin_dim1 + 1],
        //     gradin_data[i * gradin_dim1 + 2]);
        // }
        // printf("\n==================\n");

        torch_launch_calculate_nepforce_grad(
            list_neigh, 
            Ri_d, 
            grad_output[0], 
            natoms, 
            neigh_num, 
            grad);
        // std::cout << "CalculateNepForce::backward out grad shape: " << grad.sizes() << std::endl;
        // gradin = grad.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     if (i > 0) continue;
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("backforce gard[%d][%d][:] = %f %f %f %f\n", i, j, 
        //         gradin_data[i * gradin_dim1 * 4 + j * 4 + 0],
        //         gradin_data[i * gradin_dim1 * 4 + j * 4 + 1],
        //         gradin_data[i * gradin_dim1 * 4 + j * 4 + 2],
        //         gradin_data[i * gradin_dim1 * 4 + j * 4 + 3]);
        //     }
        // }
        // printf("\n==================\n");

        return {
            torch::autograd::Variable(), 
            grad,
            torch::autograd::Variable(), 
            torch::autograd::Variable()
            };
    }

torch::autograd::variable_list calculateNepForce(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Ri_d,
    at::Tensor F)
    {
        return CalculateNepForce::apply(list_neigh, dE, Ri_d, F);
    }

// the following is the code virial
torch::autograd::variable_list CalculateVirialFuncs::forward(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        int nghost = nghost_tensor.item<int>();
        auto virial = torch::zeros({batch_size, 9}, dE.options());
        auto atom_virial = torch::zeros({batch_size, natoms + nghost, 9}, dE.options());
        torch_launch_calculate_virial_force(list_neigh, dE, Rij, Ri_d, batch_size, natoms, neigh_num, virial, atom_virial, nghost);
        return {virial, atom_virial};
    }

torch::autograd::variable_list CalculateVirialFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d) 
    {
        auto dims = list_neigh.sizes();
        int batch_size = dims[0];
        int natoms = dims[1];
        int neigh_num = dims[2] * dims[3];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_virial_force_grad(list_neigh, Rij, Ri_d, grad_output[0], batch_size, natoms, neigh_num, grad);
        return {torch::autograd::Variable(), grad, torch::autograd::Variable(), torch::autograd::Variable(), torch::autograd::Variable()};
    }

torch::autograd::variable_list CalculateVirial::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        ctx->save_for_backward({list_neigh, dE, Rij, Ri_d});
        return CalculateVirialFuncs::forward(list_neigh, dE, Rij, Ri_d, nghost_tensor);
    }

torch::autograd::variable_list CalculateVirial::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Rij = saved[2];
        auto Ri_d = saved[3];
        return CalculateVirialFuncs::backward(grad_output, list_neigh, dE, Rij, Ri_d);
    }

torch::autograd::variable_list calculateVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor nghost_tensor) 
    {
        return CalculateVirial::apply(list_neigh, dE, Rij, Ri_d, nghost_tensor);
    }

// the nep virial
torch::autograd::variable_list CalculateNepVirial::forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor list_neigh,
        at::Tensor dE,
        at::Tensor Rij,
        at::Tensor Ri_d,
        at::Tensor num_atom
        )
    {
        auto dims = list_neigh.sizes();
        int natoms = dims[0];
        int neigh_num = dims[1];
        int batch_size = num_atom.sizes()[0];
        auto virial = torch::zeros({batch_size, 9}, dE.options());
        auto atom_virial = torch::zeros({natoms, 9}, dE.options());
        // printf("CalculateNepVirial forward batch_size %d\n", batch_size);
        torch_launch_calculate_nepvirial(
            list_neigh, 
            dE, 
            Rij, 
            Ri_d,
            num_atom,
            batch_size,
            natoms, 
            neigh_num, 
            virial, 
            atom_virial);
        ctx->save_for_backward({list_neigh, dE, Rij, Ri_d});
        return {virial, atom_virial};
    }

torch::autograd::variable_list CalculateNepVirial::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output)
    {
        // std::cout << "CalculateNepVirial::backward in grad_output[0] shape: " << grad_output[0].sizes() << std::endl;
        auto saved = ctx->get_saved_variables();
        auto list_neigh = saved[0];
        auto dE = saved[1];
        auto Rij = saved[2];
        auto Ri_d = saved[3];
        auto dims = list_neigh.sizes();
        int natoms = dims[0];
        int neigh_num = dims[1];
        auto grad = torch::zeros_like(dE);
        torch_launch_calculate_nepvirial_grad(
                list_neigh, 
                Rij, 
                Ri_d, 
                grad_output[0], 
                natoms, 
                neigh_num, 
                grad);

        // auto gradin = grad.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     if (i > 0) continue;
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //     printf("CalculateNepVirial back[%d][%d][:] = %f %f %f %f\n", i, j,
        //     gradin_data[i * gradin_dim1 * gradin_dim2 + j * 4 + 0],
        //     gradin_data[i * gradin_dim1 * gradin_dim2 + j * 4 + 1],
        //     gradin_data[i * gradin_dim1 * gradin_dim2 + j * 4 + 2],
        //     gradin_data[i * gradin_dim1 * gradin_dim2 + j * 4 + 3]);
        //     }
        // }
        // std::cout << "CalculateNepVirial::backward out grad shape: " << grad.sizes() << std::endl;
        return {torch::autograd::Variable(), 
                grad, 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable()
                };
    }

torch::autograd::variable_list calculateNepVirial(
    at::Tensor list_neigh,
    at::Tensor dE,
    at::Tensor Rij,
    at::Tensor Ri_d,
    at::Tensor num_atom
    )
    {
        return CalculateNepVirial::apply(list_neigh, dE, Rij, Ri_d, num_atom);
    }

// the following is the code compress
torch::autograd::variable_list CalculateCompressFuncs::forward(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        auto dims = coefficient.sizes();
        int64_t sij_num = dims[0];
        int64_t layer_node = dims[1];
        int64_t coe_num = dims[2];
        auto G = torch::empty({sij_num, layer_node}, f2.options());
        torch_launch_calculate_compress(f2, coefficient, sij_num, layer_node, coe_num, G);
        return {G};
    }

torch::autograd::variable_list CalculateCompressFuncs::backward(
    torch::autograd::variable_list grad_output,
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        auto dims = coefficient.sizes();
        int64_t sij_num = dims[0];
        int64_t layer_node = dims[1];
        int64_t coe_num = dims[2];
        auto grad = torch::zeros({sij_num, layer_node}, f2.options());
        torch_launch_calculate_compress_grad(f2, coefficient, grad_output[0], sij_num, layer_node, coe_num, grad);
        auto grad_out = torch::sum(grad_output[0]*grad, -1).unsqueeze(-1); //对应坐标位置相乘，然后算加法
        return {grad_out, torch::autograd::Variable()}; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateCompress::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        ctx->save_for_backward({f2, coefficient});
        return CalculateCompressFuncs::forward(f2, coefficient);
    }

torch::autograd::variable_list CalculateCompress::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto f2 = saved[0];
        auto coefficient = saved[1];
        return CalculateCompressFuncs::backward(grad_output, f2, coefficient);
    }

torch::autograd::variable_list calculateCompress(
    at::Tensor f2,
    at::Tensor coefficient) 
    {
        return CalculateCompress::apply(f2, coefficient);
    }

torch::autograd::variable_list CalculateNepFeat::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    at::Tensor feats,
    double rcut_radial,
    int64_t multi_feat_num,
    int64_t fix_cij) 
    {
        // printf("============ 2b forward ===========\n");
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t atom_nums = dims_image[0];
        int64_t maxneighs = dims_image[1];
        // options() 集成dtype和device
        auto dfeat_c2 = torch::zeros({atom_nums, atom_types, n_base_2b}, d12_radial.options());
        auto dfeat_2b = torch::zeros({atom_nums, maxneighs, n_max_2b}, d12_radial.options()); // [i, j, n] dfeature/drij
        auto dfeat_2b_noc = torch::zeros({atom_nums, maxneighs, n_base_2b, 4}, d12_radial.options()); // [i, j, n] dfeature/drij without c
        
        torch_launch_calculate_nepfeat(coeff2, d12_radial, NL_radial, atom_map, 
                                rcut_radial, feats, dfeat_c2, dfeat_2b, dfeat_2b_noc,
                                    atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types);
                     // 交换coeff2维度，方便在核函数中检索c下标
        ctx->save_for_backward({coeff2, d12_radial, NL_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, atom_map});
        ctx->saved_data["multi_feat_num"] = multi_feat_num;
        ctx->saved_data["fix_cij"] = fix_cij;

        return {feats};
    }

torch::autograd::variable_list CalculateNepFeat::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        // std::cout << "CalculateNepFeat::backward in grad_output 2b shape: " << grad_output[0].sizes() << std::endl;
        // auto gradin = grad_output[0].to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("2b grad_output[%ld][%ld][:] = %f %f %f %f %f\n", i, j,
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 0],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 1],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 2],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 3],
        //          gradin_data[i * gradin_dim1 * gradin_dim2  + j * gradin_dim2 + 4]);
        //         }
        //     printf("\n");
        // }
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto NL_radial = saved[2];
        auto dfeat_c2 = saved[3];
        auto dfeat_2b = saved[4];
        auto dfeat_2b_noc = saved[5];
        auto atom_map = saved[6];
        int64_t multi_feat_num = ctx->saved_data["multi_feat_num"].toInt();
        int64_t fix_cij = ctx->saved_data["fix_cij"].toInt();
        
        auto result = CalculateNepFeatGrad::apply(grad_output[0], coeff2, d12_radial, NL_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, atom_map, multi_feat_num, fix_cij);
        // return CalculateNepFeatFuncs::backward(grad_output, coeff2, d12_radial, dfeat_c2, dfeat_2b, atom_map);
        auto grad_coeff2 = result[0];
        auto grad_d12_radial = result[1];
        

        // auto gradin = grad_coeff2.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t l = 0; l < gradin_dim2; ++l) {
        //             printf("2b grad_coeff2[%ld][%ld][%ld][:] = ", i, j, l);
        //             for (int64_t p = 0; p < gradin_dim3; ++p) {
        //             printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3
        //                                     + j * gradin_dim2 * gradin_dim3 
        //                                     + l * gradin_dim3 
        //                                     + p]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("====end====");
        // gradin = grad_d12_radial.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // gradin_dim2 = gradin_shape[2];
        // gradin_dim3 = gradin_shape[3];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t l = 0; l < gradin_dim2; ++l) {
        //             if (j > 3) continue;
        //             printf("2b grad_d12_radial[%ld][%ld][%ld][:] = ", i, j, l);
        //             for (int64_t p = 0; p < gradin_dim3; ++p) {
        //             printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3
        //                                     + j * gradin_dim2 * gradin_dim3 
        //                                     + l * gradin_dim3 
        //                                     + p]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("====end====");
        // std::cout << "CalculateNepFeat::backward out grad_coeff2 2b shape: " << grad_coeff2.sizes() << std::endl;
        return {grad_coeff2, 
                grad_d12_radial, 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable()
                }; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateNepFeatGrad::forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor grad_input,
            at::Tensor coeff2,
            at::Tensor d12_radial,
            at::Tensor NL_radial,
            at::Tensor dfeat_c2,
            at::Tensor dfeat_2b,
            at::Tensor dfeat_2b_noc,
            at::Tensor atom_map,
            int64_t multi_feat_num,
            int64_t fix_cij) 
    {
        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t atom_nums = dims_image[0];
        int64_t maxneighs = dims_image[1];
        // options() 集成dtype和device
        auto grad_coeff2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, d12_radial.options());
        auto grad_d12_radial = torch::zeros({atom_nums, maxneighs, 4}, d12_radial.options());

        torch_launch_calculate_nepfeat_grad(grad_input, dfeat_c2, dfeat_2b, atom_map, 
                     atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, multi_feat_num, grad_coeff2, grad_d12_radial);
        // grad_coeff2 = grad_coeff2.permute({0, 2, 1, 3});
        ctx->save_for_backward({coeff2, d12_radial, NL_radial, dfeat_c2, dfeat_2b, dfeat_2b_noc, grad_input, atom_map});
        ctx->saved_data["multi_feat_num"] = multi_feat_num;
        ctx->saved_data["fix_cij"] = fix_cij;
        return {grad_coeff2, grad_d12_radial};

    }

torch::autograd::variable_list CalculateNepFeatGrad::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_second) 
    {
        // std::cout << "CalculateNepFeatGrad::backward in grad_second[1] 2b shape: " << grad_second[1].sizes() << " grad_second[0] 2b shape: " << grad_second[0].sizes() << std::endl;;
        auto saved = ctx->get_saved_variables();
        auto coeff2 = saved[0];
        auto d12_radial = saved[1];
        auto NL_radial = saved[2];
        auto dfeat_c2 = saved[3];
        auto dfeat_2b = saved[4];
        auto dfeat_2b_noc = saved[5];
        auto de_feat = saved[6];
        auto atom_map = saved[7];

        auto dims_2b = coeff2.sizes();
        int64_t atom_types = dims_2b[0];
        int64_t n_max_2b = dims_2b[2];
        int64_t n_base_2b = dims_2b[3];

        auto dims_image = d12_radial.sizes();
        int64_t atom_nums = dims_image[0];
        int64_t maxneighs = dims_image[1];

        int64_t multi_feat_num = ctx->saved_data["multi_feat_num"].toInt();
        int64_t fix_cij = ctx->saved_data["fix_cij"].toInt();
        // std::cout << "CalculateNepFeatGrad::backward grad_second[0] shape: " << grad_second[0].sizes() << std::endl;
        // std::cout << "CalculateNepFeatGrad::backward grad_second[1] shape: " << grad_second[1].sizes() << std::endl;
        // auto gradin = grad_second[1].to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     if (i != 10) continue;
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("2b grad_second[1][%d][%d][:] = %f %f %f %f\n", i, j,
        //         gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +0],
        //         gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +1],
        //         gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +2],
        //         gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +3]);
        //     }
        // }
        // printf("\n==================\n");

        // auto gradsecond_gradout = torch::zeros({batch_size, atom_nums, maxneighs, n_max_2b, 4}, coeff2.options());
        // gradsecond_gradout.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
        //                            torch::indexing::Slice(), 0}, dfeat_2b.squeeze(-1));
        // // 3. 逐元素乘法：C * A_expanded，得到 [batch, atom_nums, maxneighs, n_max_2b, 4]
        // gradsecond_gradout = (grad_second[1].unsqueeze(-2) * gradsecond_gradout).sum(-1).sum(-2);  // 对最后一个维度进行求和，得到 [batch, atom_nums, n_max_2b]

        // 对gradout的导数
        auto gradsecond_gradout = torch::zeros({atom_nums, n_max_2b}, coeff2.options());
        torch_launch_calculate_nepfeat_secondgradout(grad_second[1], dfeat_2b, atom_nums, maxneighs, n_max_2b, gradsecond_gradout);

        // gradin = gradsecond_gradout.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     printf("2b gradsecond_gradout[%d][:] = ", i);
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("%f ", gradin_data[i * gradin_dim1 + j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n==================\n");

        // 对C的导数
        auto gradsecond_c2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, coeff2.options());
        if (fix_cij == 0) {
        torch_launch_calculate_nepfeat_secondgradout_c2(grad_second[1], de_feat, dfeat_2b_noc, atom_map, NL_radial, 
                            atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, multi_feat_num, gradsecond_c2);
        }
        // auto gradsecond_c2 = torch::zeros({atom_types, atom_types, n_max_2b, n_base_2b}, coeff2.options());
        // gradsecond_c2 = de_feat.unsqueeze(2).unsqueeze(3).unsqueeze(5);
        // gradsecond_c2 =  gradsecond_c2 * dfeat_2b_noc.unsqueeze(4);
        // gradsecond_c2 = grad_second[1].unsqueeze(3).unsqueeze(4) * gradsecond_c2;
        // gradsecond_c2 = gradsecond_c2.sum(-1);

        // gradin = gradsecond_c2.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // printf("gradsecond_c2 shape [%d  %d  %d  %d]\n", gradin_dim0, gradin_dim1, gradin_dim2, gradin_dim3);
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         for (int64_t k = 0; k < gradin_dim2; ++k) {
        //             printf("gradsecond_c2[%ld][%ld][%ld][:] = ", i, j, k);
        //             for (int64_t l = 0; l < gradin_dim3; ++l) {
        //                     int64_t idx = i * gradin_dim1 * gradin_dim2 * gradin_dim3 + 
        //                                   j * gradin_dim2 * gradin_dim3 + 
        //                                   k * gradin_dim3 + 
        //                                   l;
        //                     printf("%f ", gradin_data[idx]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("\n==================\n\n");

        // gradsecond_c2 = gradsecond_c2.sum(0).sum(0).sum(0).unsqueeze(0).unsqueeze(0);
        // gradsecond_c2 = gradsecond_c2.permute({0, 1, 3, 2});
        // std::cout << "CalculateNepFeatGrad::backward out gradsecond_gradout 2b shape: " << gradsecond_gradout.sizes() << " gradsecond_c2 2b shape: " << gradsecond_c2.sizes() << std::endl;;
        return {
            gradsecond_gradout,
            gradsecond_c2,
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()
            };
    }

torch::autograd::variable_list calculateNepFeat(
    at::Tensor coeff2,
    at::Tensor d12_radial,
    at::Tensor NL_radial,
    at::Tensor atom_map,
    at::Tensor feats,
    double rcut_radial,
    int64_t feat_multi_nums,
    int64_t fix_cij) 
    {
        return CalculateNepFeat::apply(coeff2, d12_radial, NL_radial, atom_map, feats, rcut_radial, feat_multi_nums, fix_cij);
    }


torch::autograd::variable_list CalculateNepMbFeat::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor coeff3,
    at::Tensor d12,
    at::Tensor NL,
    at::Tensor atom_map,
    at::Tensor feats,
    int64_t feat_2b_num,
    int64_t lmax_3,
    int64_t lmax_4,
    int64_t lmax_5,
    double rcut_angular,
    int64_t fix_cij) 
    {
        // printf("============ 3b forward ===========\n");
        auto dims = coeff3.sizes();
        int64_t atom_types = dims[0];
        int64_t n_max = dims[2];
        int64_t n_base = dims[3];
        auto dims_image = d12.sizes();
        int64_t atom_nums = dims_image[0];
        int64_t maxneighs = dims_image[1];
        // options() 集成dtype和device
        const int64_t NUM_OF_ABC = 24;
        auto dfeat_c3 = torch::zeros({atom_nums, atom_types, n_base}, d12.options());
        auto dfeat_3b = torch::zeros({atom_nums, maxneighs, n_max}, d12.options()); // [i, j, n] dfeature/drij
        auto dfeat_3b_noc = torch::zeros({atom_nums, maxneighs, n_base, 4}, d12.options()); // [i, j, n] dfeature/drij without c
        auto sum_fxyz = torch::zeros({atom_nums, n_max * NUM_OF_ABC}, d12.options());
        torch_launch_calculate_nepmbfeat(coeff3, d12, NL, atom_map,
                                feats, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz,
                                    rcut_angular, atom_nums, maxneighs, n_max, n_base, lmax_3, lmax_4, lmax_5, atom_types);
                     // 交换coeff2维度，方便在核函数中检索c下标
        ctx->save_for_backward({coeff3, d12, NL, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz, atom_map});
        ctx->saved_data["rcut_angular"] = rcut_angular;
        ctx->saved_data["feat_2b_num"] = feat_2b_num;
        ctx->saved_data["lmax_3"] = lmax_3;
        ctx->saved_data["lmax_4"] = lmax_4;
        ctx->saved_data["lmax_5"] = lmax_5;
        ctx->saved_data["fix_cij"] = fix_cij;
        return {feats};
    }

torch::autograd::variable_list CalculateNepMbFeat::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_output) 
    {
        // std::cout << "CalculateNepMbFeat::backward in grad_output[0] 3b shape: " << grad_output[0].sizes() << std::endl;;
        auto saved = ctx->get_saved_variables();
        auto coeff3 = saved[0];
        auto d12 = saved[1];
        auto NL = saved[2];
        auto dfeat_c3 = saved[3];
        auto dfeat_3b = saved[4];
        auto dfeat_3b_noc = saved[5];
        auto sum_fxyz = saved[6];
        auto atom_map = saved[7];
        double rcut_angular = ctx->saved_data["rcut_angular"].toDouble();
        int64_t feat_2b_num = ctx->saved_data["feat_2b_num"].toInt();
        int64_t lmax_3 = ctx->saved_data["lmax_3"].toInt();
        int64_t lmax_4 = ctx->saved_data["lmax_4"].toInt();
        int64_t lmax_5 = ctx->saved_data["lmax_5"].toInt();
        int64_t fix_cij = ctx->saved_data["fix_cij"].toInt();
        auto result = CalculateNepMbFeatGrad::apply(grad_output[0], coeff3, d12, NL, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz, atom_map, feat_2b_num, lmax_3, lmax_4, lmax_5, rcut_angular, fix_cij);
        auto grad_coeff3 = result[0];
        auto grad_d12_angular = result[1];
        // std::cout << "CalculateNepMbFeat::backward out grad_coeff3 3b shape: " << grad_coeff3.sizes() << " grad_d12_angular 3b shape: " << grad_d12_angular.sizes() << std::endl;;
        return {grad_coeff3, 
                grad_d12_angular, 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable(), 
                torch::autograd::Variable()
                }; // f2的梯度 coeff的梯度 因为不需要，所以不计算，只占位
    }

torch::autograd::variable_list CalculateNepMbFeatGrad::forward(
            torch::autograd::AutogradContext *ctx,
            at::Tensor grad_input,
            at::Tensor coeff3,
            at::Tensor d12,
            at::Tensor NL,
            at::Tensor dfeat_c3,
            at::Tensor dfeat_3b,
            at::Tensor dfeat_3b_noc,
            at::Tensor sum_fxyz,
            at::Tensor atom_map,
            int64_t feat_2b_num,
            int64_t lmax_3,
            int64_t lmax_4,
            int64_t lmax_5,
            double rcut_angular,
            int64_t fix_cij) 
    {
        // printf("============ 3b firstgrad forward ===========\n");
        auto dims = coeff3.sizes();
        int64_t atom_types = dims[0];
        int64_t n_max = dims[2];
        int64_t n_base = dims[3];

        auto dims_image = d12.sizes();
        int64_t atom_nums = dims_image[0];
        int64_t maxneighs = dims_image[1];

        // std::cout << "CalculateNepMbFeatGrad::forward grad_input shape: " << grad_input.sizes() << std::endl;
        int64_t feat_3b_num = 0;
        int nlm = 0;
        if (lmax_3 > 0) {
            feat_3b_num += n_max * lmax_3;
            nlm += lmax_3;
        }
        if (lmax_4 > 0) {
            feat_3b_num += n_max;
            nlm += 1;
        } 
        if (lmax_5 > 0) {
            feat_3b_num += n_max;
            nlm += 1;
        }
        // options() 集成dtype和device
        const int64_t NUM_OF_ABC = 24;
        auto grad_coeff3= torch::zeros({atom_types, atom_types, n_max, n_base}, d12.options());
        auto grad_d12_angular = torch::zeros({atom_nums, maxneighs, 4}, d12.options());
        auto dsnlm_dc   = torch::zeros({atom_nums, atom_types, n_base, NUM_OF_ABC}, d12.options());
        auto dfeat_drij = torch::zeros({atom_nums, maxneighs, feat_3b_num, 4}, d12.options());
        torch_launch_calculate_nepmbfeat_grad(
            // grad_input.view({atom_nums, feat_3b_num}), 
            grad_input,
                coeff3, d12, NL, atom_map, rcut_angular, 
                    atom_nums, maxneighs, feat_2b_num, n_max, n_base, 
                        lmax_3, lmax_4, lmax_5, atom_types,
                            sum_fxyz, grad_coeff3, grad_d12_angular, dsnlm_dc, dfeat_drij);

        ctx->save_for_backward({coeff3, d12, NL, sum_fxyz, 
                            dfeat_drij, grad_input, atom_map, dsnlm_dc});
                            
        ctx->saved_data["rcut_angular"] = rcut_angular;
        ctx->saved_data["feat_2b_num"] = feat_2b_num; //for multi body feature index offset in grad_input (dEi/dfeat)
        ctx->saved_data["lmax_3"] = lmax_3;
        ctx->saved_data["lmax_4"] = lmax_4;
        ctx->saved_data["lmax_5"] = lmax_5;
        ctx->saved_data["fix_cij"] = fix_cij;
        // std::cout << "dsnlm_dc shape: " << dsnlm_dc.sizes() << std::endl;
        // auto gradin = dsnlm_dc.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         if (i ==0 or i == 30 or i == 95){
        //         for (int64_t l = 0; l < gradin_dim2; ++l) {
        //             printf("3b dsnlm_dc [i %ld][J %ld][K %ld][lm:] = ", i, j, l);
        //             for (int64_t k = 0; k < gradin_dim3; ++k) {
        //                 printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3
        //                                             + j * gradin_dim2 * gradin_dim3
        //                                             + l * gradin_dim3
        //                                             + k]);
        //                 }
        //             printf("\n");
        //         }
        //         }
        //     }
        // }

        // auto gradin = dfeat_drij.to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();         // 获取数据，假设数据是 float 类型
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // int64_t gradin_dim3 = gradin_shape[3];
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     if (i != 10) continue;
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         if (j <10 or j > 40) continue;
        //         for (int64_t l = 0; l < gradin_dim2; ++l) {
        //             printf("3b dfeat_drij [%ld][%ld][%ld][:] = ", i, j, l);
        //             for (int64_t k = 0; k < gradin_dim3; ++k) {
        //                 printf("%f ", gradin_data[i * gradin_dim1 * gradin_dim2 * gradin_dim3
        //                                         + j * gradin_dim2 * gradin_dim3
        //                                         + l * gradin_dim3
        //                                         + k]);
        //                 }
        //             printf("\n");
        //         }
        //     }
        // }
        // printf("\n==================\n");
        return {grad_coeff3, grad_d12_angular};
    }

torch::autograd::variable_list CalculateNepMbFeatGrad::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_second) 
    {
        // std::cout << "CalculateNepMbFeatGrad::backward in grad_second[1] 3b shape: " << grad_second[1].sizes() << " grad_second[0] 3b shape: " << grad_second[0].sizes() << std::endl;;
        // std::cout << "CalculateNepMbFeatGrad::backward grad_second[0] shape: " << grad_second[0].sizes() << std::endl;
        // std::cout << "CalculateNepMbFeatGrad::backward grad_second[1] shape: " << grad_second[1].sizes() << std::endl;
        // auto gradin = grad_second[1].to(torch::kCPU);  // 确保它在 CPU 上
        // auto gradin_data = gradin.data<double>();
        // auto gradin_shape = gradin.sizes();
        // int64_t gradin_dim0 = gradin_shape[0];
        // int64_t gradin_dim1 = gradin_shape[1];
        // int64_t gradin_dim2 = gradin_shape[2];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         if (i == 0) {
        //             printf("3b grad_second[1][%d][%d][:] = %f %f %f %f\n", i, j, 
        //             gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +0],
        //             gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +1],
        //             gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +2],
        //             gradin_data[i * gradin_dim1 * gradin_dim2 + j * gradin_dim2 +3]);
        //         }
        //     }
        // }
        // printf("\n==================\n");

        auto saved = ctx->get_saved_variables();
        auto coeff3 = saved[0];
        auto d12 = saved[1];
        auto NL = saved[2];
        auto sum_fxyz = saved[3];
        auto dfeat_drij = saved[4];
        auto de_feat = saved[5];
        auto atom_map = saved[6];
        auto dsnlm_dc = saved[7];
        double rcut_angular = ctx->saved_data["rcut_angular"].toDouble();
        int64_t feat_2b_num = ctx->saved_data["feat_2b_num"].toInt();
        int64_t lmax_3 = ctx->saved_data["lmax_3"].toInt();
        int64_t lmax_4 = ctx->saved_data["lmax_4"].toInt();
        int64_t lmax_5 = ctx->saved_data["lmax_5"].toInt();
        int64_t fix_cij = ctx->saved_data["fix_cij"].toInt();

        auto dims = coeff3.sizes();
        int64_t atom_types = dims[0];
        int64_t n_max = dims[2];
        int64_t n_base = dims[3];

        auto dims_image = d12.sizes();
        int64_t atom_nums = dims_image[0];
        int64_t maxneighs = dims_image[1];

        int64_t feat_3b_num = 0;
        if (lmax_3 > 0) feat_3b_num += n_max * lmax_3;
        if (lmax_4 > 0) feat_3b_num += n_max;
        if (lmax_5 > 0) feat_3b_num += n_max;

        auto gradsecond_gradout = torch::zeros({atom_nums, feat_3b_num}, coeff3.options());
        // gradsecond_gradout = (grad_second[1].unsqueeze(-2) * dfeat_drij).sum(-1).sum(-2); //Similar in function to the lower kernel function
        torch_launch_calculate_nepmbfeat_secondgradout(grad_second[1], 
                                                        dfeat_drij, 
                                                        atom_nums, 
                                                        maxneighs, 
                                                        feat_3b_num, 
                                                        gradsecond_gradout);

        // gradin = gradsecond_gradout.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // gradin_shape = gradin.sizes();
        // gradin_dim0 = gradin_shape[0];
        // gradin_dim1 = gradin_shape[1];
        // // 遍历前面三维
        // for (int64_t i = 0; i < gradin_dim0; ++i) {
        //     for (int64_t j = 0; j < gradin_dim1; ++j) {
        //         printf("3b gradsecond_gradout[%d][%d] = %f\n", i, j, 
        //         gradin_data[i * gradin_dim1 + j]);
        //     }
        // }
        // printf("\n==================\n");


        auto gradsecond_c3 = torch::zeros({atom_types, atom_types, n_max, n_base}, coeff3.options());
        if (fix_cij == 0) {
        torch_launch_calculate_nepmbfeat_secondgradout_c3(grad_second[1],
                                                            d12,
                                                            NL,
                                                            de_feat,
                                                            dsnlm_dc,
                                                            sum_fxyz,
                                                            atom_map,
                                                            coeff3,
                                                            rcut_angular,
                                                            atom_nums, 
                                                            maxneighs, 
                                                            n_max,
                                                            n_base, 
                                                            atom_types, 
                                                            lmax_3, 
                                                            lmax_4, 
                                                            lmax_5, 
                                                            feat_2b_num,
                                                            feat_3b_num, 
                                                            gradsecond_c3);
        }
        // std::cout << "CalculateNepMbFeatGrad::backward gradsecond_c3 shape: " << grad_second[1].sizes() << std::endl;
        // gradin = gradsecond_c3.to(torch::kCPU);  // 确保它在 CPU 上
        // gradin_data = gradin.data<double>();
        // for(int i = 0; i < atom_types; i++) {
        //     for(int j = 0; j < atom_types; j++) {
        //         for (int n = 0; n < n_max; n++) {
        //             printf("gradsecond_c3[i %d][J %d][n %d][k:] = ", i, j, n);
        //             for(int k = 0; k < n_base; k++) {
        //                 printf("%f ", gradin_data[i * atom_types * n_max * n_base + j * n_max * n_base + n * n_base + k]);
        //             }
        //             printf("\n");
        //         }
        //     }
        // }
        // std::cout << "CalculateNepMbFeatGrad::backward out gradsecond_gradout 3b shape: " << gradsecond_gradout.sizes() << " gradsecond_c3 3b shape: " << gradsecond_c3.sizes() << std::endl;;
        return {
            gradsecond_gradout,
            gradsecond_c3,
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(), 
            torch::autograd::Variable(), 
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()
            };
    }

torch::autograd::variable_list calculateNepMbFeat(
    at::Tensor coeff,
    at::Tensor d12,
    at::Tensor NL,
    at::Tensor atom_map,
    at::Tensor feats,
    int64_t feat_2b_num,
    int64_t lmax_3,
    int64_t lmax_4,
    int64_t lmax_5,
    double rcut,
    int64_t fix_cij) 
    {
        return CalculateNepMbFeat::apply(coeff, d12, NL, atom_map, feats, feat_2b_num, lmax_3, lmax_4, lmax_5, rcut, fix_cij);
    }

std::vector<torch::Tensor> calculate_maxneigh(
    const torch::Tensor &num_atoms,
    const torch::Tensor &box,
    const torch::Tensor &box_orig, 
    const torch::Tensor &num_cell, 
    const torch::Tensor &position,
    const double cutoff_2b,
    const double cutoff_3b,
    const int64_t atom_type_num,
    const torch::Tensor &atom_type_map,
    const bool with_type
){
    auto dtype = position.dtype();
    auto index_dtype = num_atoms.dtype();
    auto device = position.device();
    int64_t total_frames = num_atoms.sizes()[0];
    int64_t total_atoms = position.sizes()[0];
    torch::Tensor atom_num_list_sum = torch::cumsum(num_atoms, 0, torch::kInt64);
    if (with_type == false) {
        torch::Tensor NN_radial = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
        torch::Tensor NN_angular = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
        torch_launch_calculate_maxneigh(
            num_atoms,
            atom_num_list_sum,
            box,
            box_orig,
            num_cell,
            position,
            cutoff_2b,
            cutoff_3b,
            total_frames,
            total_atoms,
            NN_radial,
            NN_angular,
            atom_type_num,
            with_type,
            atom_type_map
        );
        return {NN_radial, NN_angular};
    } else {
        torch::Tensor NN_radial = torch::empty({total_atoms, atom_type_num}, torch::TensorOptions().dtype(index_dtype).device(device));
        torch::Tensor NN_angular = torch::empty({total_atoms, atom_type_num}, torch::TensorOptions().dtype(index_dtype).device(device));
        torch_launch_calculate_maxneigh(
            num_atoms,
            atom_num_list_sum,
            box,
            box_orig,
            num_cell,
            position,
            cutoff_2b,
            cutoff_3b,
            total_frames,
            total_atoms,
            NN_radial,
            NN_angular,
            atom_type_num,
            with_type,
            atom_type_map
        );
        return {NN_radial, NN_angular};
    }
    
}

std::vector<torch::Tensor> calculate_neighbor(
    const torch::Tensor &num_atoms, 
    const torch::Tensor &atom_type_map,
    const torch::Tensor &atom_types, 
    const torch::Tensor &box, 
    const torch::Tensor &box_orig, 
    const torch::Tensor &num_cell, 
    const torch::Tensor &position,
    const double cutoff_2b,
    const double cutoff_3b,
    const int64_t max_NN_radial,
    const int64_t max_NN_angular,
    const bool with_rij
){
    auto device = position.device();
    auto dtype = position.dtype();
    auto index_dtype = num_atoms.dtype();
    torch::Tensor atom_num_list_sum = torch::cumsum(num_atoms, 0, torch::kInt64);
    int64_t total_frames = num_atoms.sizes()[0];
    int64_t total_atoms = position.sizes()[0];
    
    torch::Tensor NN_radial  = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
    torch::Tensor NL_radial  = torch::full( {total_atoms, max_NN_radial}, -1, torch::TensorOptions().dtype(index_dtype).device(device));
    torch::Tensor NN_angular = torch::empty({total_atoms}, torch::TensorOptions().dtype(index_dtype).device(device));
    torch::Tensor NL_angular = torch::full( {total_atoms, max_NN_angular}, -1, torch::TensorOptions().dtype(index_dtype).device(device));

    if (!with_rij){
        torch::Tensor Ri_radial = torch::full(
            {total_atoms, max_NN_radial, 3}, 0.0, torch::TensorOptions().dtype(dtype).device(device));
        torch::Tensor Ri_angular = torch::full(
            {total_atoms, max_NN_angular,3}, 0.0, torch::TensorOptions().dtype(dtype).device(device));        
        torch_launch_calculate_neighbor(
            num_atoms,
            atom_num_list_sum, 
            atom_type_map, 
            atom_types,
            box,
            box_orig,
            num_cell, 
            position, 
            cutoff_2b,
            cutoff_3b,
            max_NN_radial,
            max_NN_angular,
            total_frames,
            total_atoms,
            NN_radial,
            NL_radial,
            NN_angular, 
            NL_angular,
            Ri_radial, 
            Ri_angular
            );
        return {NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular};
    } else {
        torch::Tensor Ri_radial = torch::full(
            {total_atoms, max_NN_radial, 4}, 0.0, torch::TensorOptions().dtype(dtype).device(device));
        torch::Tensor Ri_angular = torch::full(
            {total_atoms, max_NN_angular, 4}, 0.0, torch::TensorOptions().dtype(dtype).device(device));        
        torch_launch_calculate_neighbor(
            num_atoms,
            atom_num_list_sum, 
            atom_type_map, 
            atom_types,
            box,
            box_orig,
            num_cell, 
            position, 
            cutoff_2b,
            cutoff_3b,
            max_NN_radial,
            max_NN_angular,
            total_frames,
            total_atoms,
            NN_radial,
            NL_radial,
            NN_angular, 
            NL_angular,
            Ri_radial, 
            Ri_angular
            );
        return {NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular};
    }
}

std::vector<torch::Tensor> calculate_descriptor(
    const torch::Tensor &weight_radial,
    const torch::Tensor &weight_angular,
    const torch::Tensor &Ri_radial,
    const torch::Tensor &NL_radial,
    const torch::Tensor &atom_type_map,
    const double cutoff_radial,
    const double cutoff_angular,
    const int64_t max_NN_radial,
    const int64_t lmax_3,
    const int64_t lmax_4,
    const int64_t lmax_5
){
    auto device = Ri_radial.device();
    auto dtype = Ri_radial.dtype();
    auto dims = weight_radial.sizes();
    int64_t  num_types     = dims[0];
    int64_t  n_max_radial  = dims[2];
    int64_t  n_base_radial = dims[3];
    dims = weight_angular.sizes();
    int64_t  n_max_angular = dims[2];
    int64_t  n_base_angular= dims[3];

    int64_t total_atoms = Ri_radial.sizes()[0];
    int64_t dim_angular = 0;
        if (lmax_3 > 0) {
            dim_angular += n_max_angular * lmax_3;
        }
        if (lmax_4 > 0) {
            dim_angular += n_max_angular;
        } 
        if (lmax_5 > 0) {
            dim_angular += n_max_angular;
        }
    int64_t dim_radial = n_max_radial;
    torch::Tensor feats = torch::zeros({total_atoms, dim_radial+dim_angular}, torch::TensorOptions().dtype(dtype).device(device));
    torch_launch_calculate_descriptor(
        weight_radial, 
        weight_angular, 
        Ri_radial,
        NL_radial, 
        atom_type_map, 
        cutoff_radial, 
        cutoff_angular, 
        feats, 
        total_atoms,
        max_NN_radial, 
        n_max_radial, 
        n_base_radial, 
        n_max_angular, 
        n_base_angular, 
        lmax_3, 
        lmax_4, 
        lmax_5, 
        num_types);
    return {feats};
}