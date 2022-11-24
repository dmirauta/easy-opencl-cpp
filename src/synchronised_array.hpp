
#ifndef SYNCHRONISED_ARRAY_

    #define SYNCHRONISED_ARRAY_

    #include <iostream>
    #include <cassert>

    #include "abstract_synchronised_array.hpp"

    template<typename T>
    class SynchronisedArray : public AbstractSynchronisedArray
    {
        public:

            int buffsize;

            T* cpu_buff;
            
            SynchronisedArray(//cl_mem_flags flags, 
                                cl::Context &context,
                                int nx, int ny = 1, int nz = 1)
            {
                // mem_flags = flags;

                itemsx = nx;
                itemsy = ny;
                itemsz = nz;
                items = itemsx * itemsy * itemsz;

                buffsize = sizeof(T)*items;
                cpu_buff = new T[items];
                gpu_buff = cl::Buffer(context, CL_MEM_READ_WRITE, buffsize);
            }

            // SynchronisedArray(cl::Context &context, 
            //                     int nx, int ny = 1, int nz = 1)
            // {
            //     SynchronisedArray(CL_MEM_READ_WRITE, context, nx, ny, nz);
            // }

            ~SynchronisedArray()
            {
                delete [] cpu_buff;
            }

            void to_gpu(cl::CommandQueue &queue)
            {
                queue.enqueueWriteBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
            }

            void from_gpu(cl::CommandQueue &queue)
            {
                queue.enqueueReadBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
            }

            T& operator[](std::size_t i)
            {
                assert(i<itemsx);
                return cpu_buff[i];
            }

            T& operator[](std::size_t i, std::size_t j) // requires -std=c++23
            {
                assert(i<itemsx);
                assert(j<itemsy);
                return cpu_buff[i*itemsy + j];
            }

            T& operator[](std::size_t i, std::size_t j, std::size_t k)
            {
                assert(i<itemsx);
                assert(j<itemsy);
                assert(k<itemsz);
                return cpu_buff[ (i*itemsy + j)*itemsz + k ];
            }

    };

#endif
