
__kernel void _add(__global struct AddData *data)
{

    int i = get_global_id(0);

    data[i].c = data[i].a + data[i].b;

}

__kernel void _half(__global struct HalfData *data)
{

    int i = get_global_id(0);

    data[i].b = 0.5 * data[i].a;

}
