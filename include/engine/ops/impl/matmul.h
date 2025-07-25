#ifndef NAWAH_OPS_MATMUL_H
#define NAWAH_OPS_MATMUL_H

class Tensor;

Tensor matmul_cpu(const Tensor& a, const Tensor& b);
Tensor matmul_gpu(const Tensor& a, const Tensor& b);

#endif