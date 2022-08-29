#include <cassert>

//#include "synchronised_array.hpp"

template<typename T>
SynchronisedArray<T>::SynchronisedArray(cl::Context &context, int nx, int ny, int nz, std::string name)
{
    _name = name;
    //std::cout << name << " created\n";

    itemsx = nx;
    itemsy = ny;
    itemsz = nz;
    items = itemsx * itemsy * itemsz;

    buffsize = sizeof(T)*items;
    cpu_buff = new T[items];
    gpu_buff = cl::Buffer(context, CL_MEM_READ_WRITE, buffsize);
}

template<typename T>
SynchronisedArray<T>::~SynchronisedArray()
{
    //std::cout << _name << " destroyed\n"; // careful to not pass by copy!
                                            // starting to appreciate rusts borrow-checker here :)
    delete [] cpu_buff;
}

template<typename T>
void SynchronisedArray<T>::to_gpu(cl::CommandQueue &queue)
{
    queue.enqueueWriteBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
}

template<typename T>
void SynchronisedArray<T>::from_gpu(cl::CommandQueue &queue)
{
    queue.enqueueReadBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
}

template<typename T>
T& SynchronisedArray<T>::operator[](std::size_t i)
{
    assert(i<itemsx);
    return cpu_buff[i];
}

template<typename T>
T& SynchronisedArray<T>::operator[](std::size_t i, std::size_t j) // requires -std=c++23
{
    assert(i<itemsx);
    assert(j<itemsy);
    return cpu_buff[i*itemsy + j];
}

template<typename T>
T& SynchronisedArray<T>::operator[](std::size_t i, std::size_t j, std::size_t k)
{
    assert(i<itemsx);
    assert(j<itemsy);
    assert(k<itemsz);
    return cpu_buff[ (i*itemsy + j)*itemsz + k ];
}
