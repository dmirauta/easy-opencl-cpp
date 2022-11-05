//#include "kernelutils.c" //can let opencl bring this in or concatenate in easycl

__kernel void _add(__global AddData_t *data)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nj = get_global_size(1);
    int idx = i*Nj + j;

    data[idx].c = data[idx].a + data[idx].b;

}

__kernel void _halve_or_quarter(__global HoQData_t *data)
{

    int i = get_global_id(0);

    #ifdef HALVE_IS_QUARTER
        data[i].b = mult(data[i].a, 0.25);
    #else
        data[i].b = mult(data[i].a, 0.5);
    #endif

}
