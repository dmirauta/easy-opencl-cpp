/*

Defines the interface that easy_cl needs to be aware of (it does not need knowledge of templates)

*/

#ifndef ABSTRACT_SYNCHRONISED_ARRAY_

    #define ABSTRACT_SYNCHRONISED_ARRAY_

    #include <CL/cl.hpp> // Apple import differs, but removed for brevity

    class AbstractSynchronisedArray
    {
        public:
            int itemsx;
            int itemsy;
            int itemsz;
            int items;

            cl::Buffer gpu_buff;
            cl_mem_flags mem_flags;

            virtual void to_gpu(cl::CommandQueue &queue) = 0;

            virtual void from_gpu(cl::CommandQueue &queue) = 0;

    };

#endif