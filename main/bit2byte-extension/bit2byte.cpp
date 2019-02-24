#include <torch/torch.h>

using namespace at;

Tensor packing(Tensor src){
    //src is dim(32*-1) IntTensor
    //make sure shift just gnerates zero
    auto a = CUDA(kFloat).rand({2});
    a[0] = 1;
    a[1] = -2;
    Scalar mask_0 = Scalar(a[0]);
    Scalar mask_1 = Scalar(a[1]);
    src[0].__irshift__(mask_0);
    src[0].__iand__(mask_0);
    for(int i = 1; i < 32; i++){
        src[0].__ilshift__(mask_0);
        src[0].__iand__(mask_1);
        src[i].__irshift__(mask_0);
        src[i].__iand__(mask_0);
        src[0].__ior__(src[i]);
    }

    return {src[0]};
}

Tensor unpacking(Tensor src, Tensor dst){
    //src is dim(1*-1) IntTensor
    //dst is dim(32*-1) IntTensor(ones)
    auto a = CUDA(kFloat).rand({1});
    a[0] = 1;
    Scalar mask_0 = Scalar(a[0]);

    for(int i = 31; i >= 0; i--){
        dst[i].__iand__(src);
        dst[i].__ilshift__(mask_0);
        src.__irshift__(mask_0);
    }

    return {dst};
    //outside we should -(dst-1)
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packing", &packing, "packing");
  m.def("unpacking", &unpacking, "unpacking");
}