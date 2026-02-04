#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xstrided_view.hpp>
#include <iostream>

int main() {
    xt::xarray<double> a = xt::ones<double>({50, 50});
    
    auto sv = xt::strided_view(a, {xt::range(0, 1), xt::range(0, 3, 2)});
    
    std::cout << "Original array:" << std::endl;
    std::cout << "  a.data() = " << (void*)a.data() << std::endl;
    std::cout << "  a.strides() = (" << a.strides()[0] << ", " << a.strides()[1] << ")" << std::endl;
    
    std::cout << "\nStrided view:" << std::endl;
    std::cout << "  sv.shape() = (" << sv.shape()[0] << ", " << sv.shape()[1] << ")" << std::endl;
    std::cout << "  sv.strides() = (" << sv.strides()[0] << ", " << sv.strides()[1] << ")" << std::endl;
    std::cout << "  &*sv.begin() = " << (void*)&(*sv.begin()) << std::endl;
    std::cout << "  sv.data() + sv.data_offset() = " << (void*)(sv.data() + sv.data_offset()) << std::endl;
    
    std::cout << "\nValues:" << std::endl;
    std::cout << "  *(&*sv.begin()) = " << *(&*sv.begin()) << std::endl;
    std::cout << "  sv(0,0) = " << sv(0,0) << std::endl;
    std::cout << "  sv(0,1) = " << sv(0,1) << std::endl;
    
    // Address calculations
    double* base = &(*sv.begin());
    std::cout << "\nAddress calculations:" << std::endl;
    std::cout << "  base = " << (void*)base << std::endl;
    std::cout << "  base + 2 (elem stride) = " << (void*)(base + 2) << " value=" << *(base + 2) << std::endl;
    
    // The second element should be at base + stride[1] elements
    std::cout << "  base + sv.strides()[1] = " << (void*)(base + sv.strides()[1]) << " value=" << *(base + sv.strides()[1]) << std::endl;
    
    return 0;
}
