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
        int items;
        int buffsize;
        T* cpu_buff;
        cl::Buffer gpu_buff;

        std::string _name;

        SharedArray(int n, cl::Context &context, std::string name="arr")
        {
            _name = name;
            std::cout << name << " created\n";

            items = n;
            buffsize = sizeof(T)*items;
            cpu_buff = new T[items];
            gpu_buff = cl::Buffer(context, CL_MEM_READ_WRITE, buffsize);
        };

        ~SharedArray()
        {
            std::cout << _name << " destroyed\n";

            delete [] cpu_buff;
        };

        void to_gpu(cl::CommandQueue &queue)
        {
            queue.enqueueWriteBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
        };

        void from_gpu(cl::CommandQueue &queue)
        {
            queue.enqueueReadBuffer(gpu_buff, CL_TRUE, 0, buffsize, cpu_buff);
        };
//    private:
};

template<typename T>
void run_vec_kernel(cl::Context &context,
                    cl::CommandQueue &queue,
                    cl::Kernel &vec_kernel,
                    std::vector<std::reference_wrapper<SharedArray<T>>> params)
{

    for(int i=0; i<params.size(); i++)
    {
        params[i].get().to_gpu(queue);
        vec_kernel.setArg(i, params[i].get().gpu_buff);
    }

    queue.enqueueNDRangeKernel(vec_kernel, cl::NullRange, cl::NDRange(NUM_GLOBAL_WITEMS), cl::NDRange(32));

    for(auto &par : params)
    {
        par.get().from_gpu(queue);
    }

    queue.finish(); // blocking

}


int main(int argc, char* argv[]) {

    //std::cout << __LINE__ << "\n";

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
    SharedArray<int> A = SharedArray<int>(n, context, "A");
    SharedArray<int> B = SharedArray<int>(n, context, "B");
    SharedArray<int> C = SharedArray<int>(n, context, "C");

    for (int i=0; i<n; i++) {
        A.cpu_buff[i] = i;
        B.cpu_buff[i] = n - i - 1;
    }

    std::vector<std::reference_wrapper<SharedArray<int>>> params{A,B,C};

    run_vec_kernel(context, queue, kernels["vector_add"], params);

    std::cout << "\n";
    for(int i=0; i<10; i++)
    {
        std::cout << A.cpu_buff[i] << " + " << B.cpu_buff[i] << " = " << C.cpu_buff[i] << "\n";
    }

    run_vec_kernel(context, queue, kernels["vector_mult"], params);

    std::cout << "\n";
    for(int i=0; i<10; i++)
    {
        std::cout << A.cpu_buff[i] << " * " << B.cpu_buff[i] << " = " << C.cpu_buff[i] << "\n";
    }

    return 0;
}
