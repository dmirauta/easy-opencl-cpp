
__kernel void _add(__global struct AddData *data)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int Nj = get_global_size(1);
    int idx = i*Nj + j;

    data[idx].c = data[idx].a + data[idx].b;

}

__kernel void _half(__global struct HalfData *data)
{

    int i = get_global_id(0);

    data[i].b = 0.5 * data[i].a;

}
