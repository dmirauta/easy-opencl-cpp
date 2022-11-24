#include <iostream>
#include <filesystem>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "easy_cl.hpp"

#include "mandelstructs.h"

using namespace std;

int main(int argc, char* argv[]) {

    bool verbose;
    if (argc > 1 && strcmp(argv[1], "1") == 0)
        verbose = true;
    else
        verbose = false;

    EasyCL ecl(verbose);

    vector<string> source_files{"mandel.cl"};
    vector<string> kernel_names{"escape_iter"};
    string ocl_include = "-I ";
    string path = filesystem::current_path();
    ecl.load_kernels(source_files, kernel_names, ocl_include+path);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Mandelbrot

    // 4K causes segfault?
    const int N = 1080;
    const int M = 1920;

    // Setup data
    SynchronisedArray<int>     iters(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    SynchronisedArray<EIParam> param(ecl.context);

    param[0].mandel = 1;
    param[0].view_rect = {-2, 0.5, -1, 1};
    param[0].MAXITER = 1000;

    // Run kernel
    ecl.apply_kernel("escape_iter", iters, param);

    unsigned char pix[N*M*3];
    float q, it;
    float fltmax = param[0].MAXITER;
    for (int i=0; i<N*M; i++)
    {
        it = iters.cpu_buff[i];
        q = it/fltmax;
        pix[3*i]   = 255*q;
        pix[3*i+1] = 255*q;
        pix[3*i+2] = 255*q;
    }

    stbi_write_png("mandel.png", M, N, 3, pix, 3*M);
    
    return 0;
}
