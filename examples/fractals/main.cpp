#include <iostream>
#include <filesystem>
#include <algorithm>
#include <ctime>

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

    vector<string> source_files{"mandel.cl"}; // this time importing its own deps by setting include path
    vector<string> kernel_names{"escape_iter", "min_prox", "orbit_trap", "map_img", "apply_log_int", "apply_log_fpn"};
    string ocl_include = "-I ";
    string path = filesystem::current_path();
    ecl.load_kernels(source_files, kernel_names, ocl_include+path);

    int N = 2160;
    int M = 3840;
    unsigned char * pix = new unsigned char[N*M*3];

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Mandelbrot escape iters image

    // Setup data
    SynchronisedArray<int>     iters(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    SynchronisedArray<EIParam> param(ecl.context);

    param[0].mandel = 1;
    param[0].view_rect = {-2, 0.5, -1, 1};
    param[0].MAXITER = 1000;

    // Run kernel(s)
    ecl.apply_kernel("escape_iter", iters, param);
    ecl.apply_kernel("apply_log_int", iters);

    float q, it;
    float fltmax = *max_element(iters.cpu_buff, iters.cpu_buff+iters.items);
    for (int i=0; i<N*M; i++)
    {
        it = iters.cpu_buff[i];
        q = it/fltmax;
        pix[3*i]   = 255*q;
        pix[3*i+1] = 255*q;
        pix[3*i+2] = 255*q;
    }

    stbi_write_png("escape_iter.png", M, N, 3, pix, 3*M); // the time consuming bit!

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Mandelbrot min prox image

    // Setup data
    SynchronisedArray<double>  prox1(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    SynchronisedArray<double>  prox2(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    SynchronisedArray<double>  prox3(ecl.context, CL_MEM_WRITE_ONLY, {N, M});
    SynchronisedArray<MPParam> param2(ecl.context);

    param2[0].mandel = 1;
    param2[0].view_rect = {-2, 0.5, -1, 1};
    param2[0].MAXITER = 100;

    // Run kernel(s)
    param2[0].PROXTYPE = 1;
    ecl.apply_kernel("min_prox", prox1, param2);
    // ecl.apply_kernel("apply_log_fpn", prox1);

    param2[0].PROXTYPE = 2;
    ecl.apply_kernel("min_prox", prox2, param2);
    // ecl.apply_kernel("apply_log_fpn", prox2);

    param2[0].PROXTYPE = 4;
    ecl.apply_kernel("min_prox", prox3, param2);
    // ecl.apply_kernel("apply_log_fpn", prox3);

    // double p1max = *max_element(prox1.cpu_buff, prox1.cpu_buff+prox1.items);
    // double p2max = *max_element(prox2.cpu_buff, prox2.cpu_buff+prox1.items);
    // double p3max = *max_element(prox3.cpu_buff, prox3.cpu_buff+prox1.items);
    double s;
    for (int i=0; i<N*M; i++)
    {
        s = prox1.cpu_buff[i] + prox2.cpu_buff[i] + prox3.cpu_buff[i];
        pix[3*i]   = 255*(prox1.cpu_buff[i]/s);
        pix[3*i+1] = 255*(prox2.cpu_buff[i]/s);
        pix[3*i+2] = 255*(prox3.cpu_buff[i]/s);
    }

    stbi_write_png("min_prox.png", M, N, 3, pix, 3*M);
    
    delete [] pix;

    return 0;
}
