#include <iostream>
#include <fstream>
#include <sstream>

//#include "quick_cl.hpp"

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

void setup_cl(cl::Context &context,
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

std::map<std::string, cl::Kernel> setup_cl_prog(cl::Context &context,
                                                cl::Device &device,
                                                std::vector<std::string> source_files,
                                                std::vector<std::string> kernel_names,
                                                std::string build_options,
                                                bool verbose)
{
    cl::Program::Sources sources;

//     // Not sure how this is supposed to work
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
                  << build_options << "\n";
        std::cout << "Source:\n"
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

template<typename T>
void apply_kernel(cl::Context &context,
                  cl::CommandQueue &queue,
                  cl::Kernel &kernel,
                  SynchronisedArray<T> &data)
{

    cl::NDRange global_dims;
    if (data.itemsz>1)
    {
        global_dims = cl::NDRange(data.itemsx, data.itemsy, data.itemsz);
    } else if (data.itemsy>1) {
        global_dims = cl::NDRange(data.itemsx, data.itemsy);
    } else if (data.itemsx>1) {
        global_dims = cl::NDRange(data.itemsx);
    } else {
        std::cout << "Invalid global dims in apply_kernel? (based on input data)\n";
        exit(1);
    }

    data.to_gpu(queue);

    kernel.setArg(0, data.gpu_buff);

    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,  // offset
                               global_dims,
                               cl::NullRange); // local  dims (warps/workgroups)

    data.from_gpu(queue);

    queue.finish(); // blocking

}
