
#ifndef EASY_CL_

    #define EASY_CL_

    #include <map>

    #include <CL/cl.hpp> // Apple import differs, but removed for brevity

    #include "abstract_synchronised_array.hpp"

    cl::NDRange get_global_dims(AbstractSynchronisedArray &arr);

    inline void to_gpu(cl::CommandQueue &queue, cl::Kernel &kernel, int firstargnum) { }

    template<typename... ASArrays>
    void to_gpu(cl::CommandQueue &queue,
                cl::Kernel &kernel,
                int firstargnum,
                AbstractSynchronisedArray& first_arr,
                ASArrays&... arrs)
    {
        first_arr.to_gpu(queue);
        kernel.setArg(firstargnum, first_arr.gpu_buff);
        to_gpu(queue, kernel, firstargnum+1, arrs...);
    }


    inline void from_gpu(cl::CommandQueue &queue) { }

    template<typename... ASArrays>
    void from_gpu(cl::CommandQueue &queue,
                AbstractSynchronisedArray& first_arr,
                ASArrays&... arrs)
    {
        first_arr.from_gpu(queue);
        from_gpu(queue, arrs...);
    }


    class EasyCL
    {
        public:
            cl::Context context;
            cl::Device device;
            cl::CommandQueue queue;

            std::map<std::string, cl::Kernel> kernels;

            bool _verbose;

            EasyCL(bool verbose=false);

            void load_kernels(std::vector<std::string> source_files,
                              std::vector<std::string> kernel_names,
                              std::string build_options);

            template<typename... ASArrays>
            void apply_kernel(std::string kernel_name, 
                              AbstractSynchronisedArray& first_arr, // only used for dims here
                              ASArrays&... arrs)
            {
                to_gpu(queue, kernels[kernel_name], 0, first_arr, arrs...);

                queue.enqueueNDRangeKernel(kernels[kernel_name],
                                        cl::NullRange,  // offset
                                        get_global_dims(first_arr),
                                        cl::NullRange); // local  dims (warps/workgroups)

                from_gpu(queue, first_arr, arrs...);

                queue.finish(); // blocking
            }


    };

#endif
