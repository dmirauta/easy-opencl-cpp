// unnecesarily copying some data to and from by packing like this, but its a simple way to handle many input/output types
// we could also separately handle structs of "Input" (not copied back), "Passedthrough" and "Output" (not copied to)?
// could also add Param struct for shared kernel args
// also some concern for GPU packing structs differently?

#ifndef SYNCHRONISED_ARRAY_

    #define SYNCHRONISED_ARRAY_

    #include <iostream>

    template<typename T>
    class SynchronisedArray
    {
        public:
            int itemsx;
            int itemsy;
            int itemsz;
            int items;

            int buffsize;

            T* cpu_buff;
            cl::Buffer gpu_buff;

            std::string _name;

            SynchronisedArray(cl::Context &context, int nx, int ny = 1, int nz = 1, std::string name="arr");

            ~SynchronisedArray();

            void to_gpu(cl::CommandQueue &queue);

            void from_gpu(cl::CommandQueue &queue);

            T& operator[](std::size_t i);
            T& operator[](std::size_t i, std::size_t j); // requires -std=c++23
            T& operator[](std::size_t i, std::size_t j, std::size_t k);
    };

    #include "synchronised_array.cpp" // to ensure compilation of relevant T's in including files

#endif
