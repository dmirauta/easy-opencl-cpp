 
struct Data
{
    int a;
    int b;
    int c;
};

__kernel void add(__global struct Data *data)
{

    int i = get_global_id(0);

    data[i].c = data[i].a + data[i].b;

}

__kernel void mult(__global struct Data *data)
{

    int i = get_global_id(0);

    data[i].c = data[i].a * data[i].b;

}
