//#include "kernelutils.c" //can either let opencl bring this in or concatenate in easycl by including in source files

__kernel void mult_add(__global MultAddIn_t  *indata,
                       __global MultAddOut_t *outdata )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nj = get_global_size(1);
    int idx = i*Nj + j;

    outdata[idx].add  = indata[idx].in1 + indata[idx].in2;
    outdata[idx].mult = indata[idx].in1 * indata[idx].in2;

}

__kernel void halve_or_quarter(__global HoQData_t *data)
{

    int i = get_global_id(0);

    #ifdef DO_QUARTER
        data[i].out = mult(data[i].in, 0.25);
    #else
        data[i].out = mult(data[i].in, 0.5);
    #endif

}
