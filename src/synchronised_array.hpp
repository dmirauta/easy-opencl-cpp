
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
            
            SynchronisedArray(cl::Context &context, cl_mem_flags flags, Dims dimensions)
            {
                mem_flags = flags;

                dims = dimensions;
                items = dims.x * dims.y * dims.z;

                buffsize = sizeof(T)*items;
                cpu_buff = new T[items];
                gpu_buff = cl::Buffer(context, flags, buffsize);
            }

            SynchronisedArray(cl::Context &context, Dims dimensions) 
                : SynchronisedArray(context, CL_MEM_READ_WRITE, dimensions) {}

            ~SynchronisedArray()
            {
                delete [] cpu_buff;
            }

            void to_gpu(cl::CommandQueue &queue)
            {
                if (mem_flags!=CL_MEM_WRITE_ONLY) // gpu will not need to read it, no need to copy to
                    queue.enqueueWriteBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
            }

            void from_gpu(cl::CommandQueue &queue)
            {
                if (mem_flags!=CL_MEM_READ_ONLY) // gpu will not write to it, no need to bring it back
                    queue.enqueueReadBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
            }

            T& operator[](std::size_t i)
            {
                assert(i<dims.x);
                return cpu_buff[i];
            }

            T& operator[](std::size_t i, std::size_t j) // requires -std=c++23
            {
                assert(i<dims.x);
                assert(j<dims.y);
                return cpu_buff[i*dims.y + j];
            }

            T& operator[](std::size_t i, std::size_t j, std::size_t k)
            {
                assert(i<dims.x);
                assert(j<dims.y);
                assert(k<dims.z);
                return cpu_buff[ (i*dims.y + j)*dims.z + k ];
            }

    };

#endif
