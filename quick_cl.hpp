// unnecesarily copying some data to and from by packing like this, but its a simple way to handle many input/output types
// we could also separately handle structs of "Input" (not copied back), "Passedthrough" and "Output" (not copied to)?
// could also add Param struct for shared kernel args
// also some concern for GPU packing structs differently?

#ifndef QUICK_CL_

    #define QUICK_CL_

    #include <map>

    #include <CL/cl.hpp> // Apple import differs, but removed for brevity

    #include "synchronised_array.hpp"

    std::string read_string_from_file(const std::string &file_path);

    void setup_cl(cl::Context &context,
                  cl::Device &device,
                  cl::CommandQueue &queue,
                  bool verbose = false);

    std::map<std::string, cl::Kernel> setup_cl_prog(cl::Context &context,
                                                    cl::Device &device,
                                                    std::vector<std::string> source_files,
                                                    std::vector<std::string> kernel_names,
                                                    std::string build_options,
                                                    bool verbose = false);

    template<typename T>
    void apply_kernel(cl::Context &context,
                    cl::CommandQueue &queue,
                    cl::Kernel &kernel,
                    SynchronisedArray<T> &data);

    #include "quick_cl.cpp"

#endif
