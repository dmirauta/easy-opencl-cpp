/*

Defines the interface that easy_cl needs to be aware of (it does not need knowledge of templates)

*/

#ifndef ABSTRACT_SYNCHRONISED_ARRAY_

    #define ABSTRACT_SYNCHRONISED_ARRAY_

    #include <CL/cl.hpp> // Apple import differs, but removed for brevity

    class Dims
    {
        public:
            int x;
            int y;
            int z;

            Dims(int nx=1, int ny=1, int nz=1)
            {
                x = nx;
                y = ny;
                z = nz;
            }

            Dims(std::initializer_list<int> args)
            {

                if (args.size()>=1) {
                    x = args.begin()[0];
                } else {
                    x = 1;
                }

                if (args.size()>=2) {
                    y = args.begin()[1];
                } else {
                    y = 1;
                }

                if (args.size()>=3) {
                    z = args.begin()[2];
                } else {
                    z = 1;
                }

            }

            void operator=(const Dims& d)
            {
                x=d.x;
                y=d.y;
                z=d.z;
            }


    };

    class AbstractSynchronisedArray
    {
        public:
            Dims dims;
            int items;

            cl::Buffer gpu_buff;
            cl_mem_flags mem_flags;

            virtual void to_gpu(cl::CommandQueue &queue) = 0;

            virtual void from_gpu(cl::CommandQueue &queue) = 0;

    };

#endif