
__kernel void _add(__global AddData_t *data)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nj = get_global_size(1);
    int idx = i*Nj + j;

    data[idx].c = data[idx].a + data[idx].b;

}

__kernel void _halve(__global HalveData_t *data)
{

    int i = get_global_id(0);

    #ifdef HALVE_IS_QUARTER
    data[i].b = 0.25 * data[i].a;
    #else
    data[i].b = 0.5 * data[i].a;
    #endif

}
