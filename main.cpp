#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#define NUM_GLOBAL_WITEMS 1024


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

// file mostly based on https://github.com/Dakkers/OpenCL-examples/blob/master/example01/main.cpp

void setup_cl(cl::Context &context,
              cl::Device &device,
              cl::CommandQueue &queue)
{
     // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    // std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    device=all_devices[0];
    std::cout<< "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";
    context = cl::Context({device});
    queue = cl::CommandQueue(context, device);
}

std::map<std::string, cl::Kernel> setup_cl_prog(cl::Context &context,
                                                cl::Device &device,
                                                std::vector<std::string> source_files,
                                                std::vector<std::string> kernel_names)
{
    cl::Program::Sources sources;

    std::string kernel_code;
    for(auto source_file : source_files)
    {
        kernel_code = read_string_from_file(source_file);
        sources.push_back({kernel_code.c_str(), kernel_code.length()});
    }


    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
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
class SharedArray
{
    public:
        SharedArray(int n);

        void to_gpu();
        void from_gpu();

        int size;
        T* cpu_arr;
        cl::Buffer gpu_arr;

//    private:
}

SharedArray::SharedArray(int n, cl::Context &context)
{
    size = n;
    gpu_arr(context, CL_MEM_READ_WRITE, sizeof(T)*n)
}

void SharedArray::to_gpu()
{

}

void SharedArray::from_gpu()
{

}

void run_vec_kernel(cl::Context &context,
                    cl::CommandQueue &queue,
                    cl::Kernel &vec_kernel,
                    int *A, int *B, int *C, int n)
{

    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int)*n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int)*n);

    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);

    vec_kernel.setArg(0, buffer_A);
    vec_kernel.setArg(1, buffer_B);
    vec_kernel.setArg(2, buffer_C);

    queue.enqueueNDRangeKernel(vec_kernel, cl::NullRange, cl::NDRange(NUM_GLOBAL_WITEMS), cl::NDRange(32));

    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C);

    queue.finish(); // blocking

}


int main(int argc, char* argv[]) {

    bool verbose;
    if (argc == 1 || std::strcmp(argv[1], "0") == 0)
        verbose = true;
    else
        verbose = false;

    const int n = 8*32*512;             // size of vectors

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    setup_cl(context, device, queue);

    std::vector<std::string> source_files{"vector_ops_kernel.cl"};
    std::vector<std::string> kernel_names{"vector_add", "vector_mult"};
    std::map<std::string, cl::Kernel> kernels = setup_cl_prog(context, device, source_files, kernel_names);

    // construct vectors
    int A[n], B[n], C[n];
    for (int i=0; i<n; i++) {
        A[i] = i;
        B[i] = n - i - 1;
    }

    std::cout << "\n";
    run_vec_kernel(context, queue, kernels["vector_add"], A, B, C, n);
    for(int i=0; i<10; i++)
    {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << "\n";
    }

    std::cout << "\n";
    run_vec_kernel(context, queue, kernels["vector_mult"], A, B, C, n);
    for(int i=0; i<10; i++)
    {
        std::cout << A[i] << " * " << B[i] << " = " << C[i] << "\n";
    }

    return 0;
}
