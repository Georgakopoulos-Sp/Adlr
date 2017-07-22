#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
__global__ void ADLRUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
template <typename Dtype>
void ADLR_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  ADLRUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void ADLR_update_gpu<float>(int, float*, float*, float, float);
template void ADLR_update_gpu<double>(int, double*, double*, double, double);


}  // namespace caffe
