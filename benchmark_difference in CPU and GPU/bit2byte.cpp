#include <torch/torch.h>

using namespace at;

Tensor packing(Tensor src){
    //src is dim(32*-1) IntTensor
    //make sure shift just gnerates zero
    src[0].__irshift__(31);
    for(int i = 1; i < 32; i++){
        src[0].__ilshift__(1);
        src[i].__irshift__(31);
        src[0].__ior__(src[i]);
    }

    return {src[0]};
}

Tensor unpacking(Tensor src, Tensor dst){
    //src is dim(1*-1) IntTensor
    //dst is dim(32*-1) IntTensor(ones)

    for(int i = 31; i >= 0; i--){
        dst[i].__iand__(src);
        dst[i].__ilshift__(1);
        src.__irshift__(1);
    }

    return {dst};
    //outside we should -(dst-1)
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packing", &packing, "packing");
  m.def("unpacking", &unpacking, "unpacking");
}