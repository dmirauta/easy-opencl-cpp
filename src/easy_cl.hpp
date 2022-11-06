
#ifndef EASY_CL_

    #define EASY_CL_

    #include <map>

    #include <CL/cl.hpp> // Apple import differs, but removed for brevity

    #include "abstract_synchronised_array.hpp"

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

            void apply_kernel(std::string kernel_name, 
                              AbstractSynchronisedArray &data);

    };

#endif
