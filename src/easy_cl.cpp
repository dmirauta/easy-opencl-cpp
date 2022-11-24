#include <iostream>
#include <fstream>
#include <sstream>

#include "easy_cl.hpp"

// https://stackoverflow.com/a/62772405
std::string read_string_from_file(const std::string &file_path) {
    const std::ifstream input_stream(file_path, std::ios_base::binary);

    if (input_stream.fail()) {
        throw std::runtime_error("Failed to open file");
    }

    std::stringstream buffer;
    buffer << input_stream.rdbuf();

    return buffer.str();
}
// file otherwise mostly based on https://github.com/Dakkers/OpenCL-examples/blob/master/example01/main.cpp

void setup_ocl(cl::Context &context,
               cl::Device &device,
               cl::CommandQueue &queue,
               bool verbose)
{
     // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    device=all_devices[0];
    context = cl::Context({device});
    queue = cl::CommandQueue(context, device);

    if (verbose)
    {
        std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
        std::cout << "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";
    }
}

std::map<std::string, cl::Kernel> setup_ocl_prog(cl::Context &context,
                                                 cl::Device &device,
                                                 std::vector<std::string> source_files,
                                                 std::vector<std::string> kernel_names,
                                                 std::string build_options,
                                                 bool verbose)
{
    cl::Program::Sources sources;

//     // Not sure how multiple sources are actually supposed to be provided, we will just concatenate
//     std::string kernel_code;
//     for(auto source_file : source_files)
//     {
//         kernel_code = read_string_from_file(source_file);
//         sources.push_back({kernel_code.c_str(), kernel_code.length()});
//     }

    std::string kernel_code = "";
    for(auto source_file : source_files)
    {
        kernel_code += read_string_from_file(source_file);
    }
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    if (verbose)
    {
        std::cout << "Build options:\n"
                  << build_options << "\n"
                  << "Source:\n"
                  << kernel_code;
    }

    cl::Program program(context, sources);
    if (program.build({device}, build_options.c_str()) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    std::map<std::string, cl::Kernel> kernels;
    for(auto kernel_name : kernel_names)
    {
        kernels[kernel_name] = cl::Kernel(program, kernel_name.c_str());
    }

    return kernels;
}

cl::NDRange get_global_dims(AbstractSynchronisedArray &arr)
{
    cl::NDRange global_dims;
    if (arr.dims.z>1)
    {
        global_dims = cl::NDRange(arr.dims.x, arr.dims.y, arr.dims.z);
    } else if (arr.dims.y>1) {
        global_dims = cl::NDRange(arr.dims.x, arr.dims.y);
    } else if (arr.dims.x>1) {
        global_dims = cl::NDRange(arr.dims.x);
    } else {
        std::cout << "Invalid global dims in apply_kernel? (based on input data)\n";
        exit(1);
    }
    return global_dims;
}

void to_gpu(cl::CommandQueue &queue,
            cl::Kernel &kernel,
            int firstargnum)
{
}

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


void from_gpu(cl::CommandQueue &queue)
{
}

template<typename... ASArrays>
void from_gpu(cl::CommandQueue &queue,
              AbstractSynchronisedArray& first_arr,
              ASArrays&... arrs)
{
    first_arr.from_gpu(queue);
    from_gpu(queue, arrs...);
}

template<typename... ASArrays>
void apply_ocl_kernel(cl::CommandQueue &queue,
                      cl::Kernel &kernel,
                      AbstractSynchronisedArray& first_arr,
                      ASArrays&... arrs)
{
    to_gpu(queue, kernel, 0, first_arr, arrs...);

    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,  // offset
                               get_global_dims(first_arr),
                               cl::NullRange); // local  dims (warps/workgroups)

    from_gpu(queue, first_arr, arrs...);

}

EasyCL::EasyCL(bool verbose)
{
    _verbose = verbose;
    setup_ocl(context, device, queue, verbose);
}

void EasyCL::load_kernels(std::vector<std::string> source_files,
                          std::vector<std::string> kernel_names,
                          std::string build_options)
{
    std::map<std::string, cl::Kernel> new_kernels = setup_ocl_prog(context, device, source_files, kernel_names, build_options, _verbose);

    for (auto pair: new_kernels)
    {
        kernels[pair.first] = pair.second;
    }
}

void EasyCL::apply_kernel(std::string kernel_name, AbstractSynchronisedArray &arr)
{
    apply_ocl_kernel(queue, kernels[kernel_name], arr);

    queue.finish();
}

template<typename... ASArrays>
void EasyCL::apply_kernel(std::string kernel_name, 
                            AbstractSynchronisedArray &first_arr,
                            ASArrays&... arrs)
{
    apply_ocl_kernel(queue, kernels[kernel_name], first_arr, arrs...);

    queue.finish(); // blocking
}
