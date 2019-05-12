#include <iostream>
#include <torch/extension.h>

torch::Tensor packing(torch::Tensor src)
{
    //src is dim(32*-1) IntTensor
    //make sure shift just gnerates zero

    auto options = torch::TensorOptions()
                       .dtype(torch::kInt)
                       .device(src.device());
    torch::Tensor a = torch::zeros(2, options);
    a[0] = 1;
    a[1] = -2;
    auto mask_0 = a[0];
    auto mask_1 = a[1];

    src[0].__irshift__(mask_0);
    src[0].__iand__(mask_0);

    for (int i = 1; i < 32; i++)
    {
        src[0].__ilshift__(mask_0);
        src[0].__iand__(mask_1);
        src[i].__irshift__(mask_0);
        src[i].__iand__(mask_0);
        src[0].__ior__(src[i]);
    }

    return {src[0]};
}

torch::Tensor unpacking(torch::Tensor src, torch::Tensor dst)
{
    //src is dim(1*-1) IntTensor
    //dst is dim(32*-1) IntTensor(ones)
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt)
                       .device(src.device());
    torch::Tensor a = torch::zeros(1, options);
    a[0] = 1;
    auto mask_0 = a[0];

    for (int i = 31; i >= 0; i--)
    {
        dst[i].__iand__(src);
        dst[i].__ilshift__(mask_0);
        src.__irshift__(mask_0);
    }

    return {dst};
    //outside we should -(dst-1)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("packing", &packing, "packing");
    m.def("unpacking", &unpacking, "unpacking");
}
