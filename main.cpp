#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cassert>

#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "datastructs.c"


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
              cl::CommandQueue &queue,
              bool verbose = false)
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
                                                bool verbose = false)
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
        std::cout << "Source:\n"
                  << kernel_code;
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

        SynchronisedArray(cl::Context &context,
                    int nx,
                    int ny = 1,
                    int nz = 1,
                    std::string name="arr")
        {
            _name = name;
            //std::cout << name << " created\n";

            itemsx = nx;
            itemsy = ny;
            itemsz = nz;
            items = itemsx * itemsy * itemsz;

            buffsize = sizeof(T)*items;
            cpu_buff = new T[items];
            gpu_buff = cl::Buffer(context, CL_MEM_READ_WRITE, buffsize);
        };

        ~SynchronisedArray()
        {
            //std::cout << _name << " destroyed\n"; // careful to not pass by copy!
                                                    // starting to appreciate rusts borrow-checker here :)
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

        T& operator[](std::size_t i)
        {
            assert(i<itemsx);
            return cpu_buff[i];
        }

        T& operator[](std::size_t i, std::size_t j) // requires -std=c++23
        {
            assert(i<itemsx);
            assert(j<itemsy);
            return cpu_buff[i*itemsy + j];
        }

        T& operator[](std::size_t i, std::size_t j, std::size_t k)
        {
            assert(i<itemsx);
            assert(j<itemsy);
            assert(k<itemsz);
            return cpu_buff[ (i*itemsy + j)*itemsz + k ];
        }


//    private:
};

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

// unnecesarily copying some data to and from by packing like this, but its a simple way to handle many input/output types
// we could also separately handle structs of "Input" (not copied back), "Passedthrough" and "Output" (not copied to)?
// could also add Param struct for shared kernel args
// also some concern for GPU packing structs differently?


int main(int argc, char* argv[]) {

    //std::cout << "Still working on line " << __LINE__ << "!\n";

    bool verbose;
    if (argc > 1 && std::strcmp(argv[1], "1") == 0)
        verbose = true;
    else
        verbose = false;

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    setup_cl(context, device, queue, verbose);

    std::vector<std::string> source_files{"datastructs.c", "cast.cl"};
    std::vector<std::string> kernel_names{"_add", "_half"};
    std::map<std::string, cl::Kernel> kernels = setup_cl_prog(context, device, source_files, kernel_names, verbose);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Adding test

    const int m1 = 1024;
    const int m2 = 768;
    const int _m_preview = 3;

    // Setup data
    SynchronisedArray<AddData> adddata = SynchronisedArray<AddData>(context, m1, m2);
    for (int i=0; i<m1; i++)
    {
        for (int j=0; j<m2; j++)
        {
            adddata[i, j].a = i;
            adddata[i, j].b = j;
        }
    }

    // Run kernel
    apply_kernel(context, queue, kernels["_add"], adddata);
    // Preview results
    std::cout << "\n" << "Adding (last "<<_m_preview<<"x"<<_m_preview<<")\n";
    for(int i=m1-_m_preview; i<m1; i++)
    {
        for(int j=m2-_m_preview; j<m2; j++)
        {
            std::cout << adddata[i, j].a << " + "
                      << adddata[i, j].b << " = "
                      << adddata[i, j].c << "\n";
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Halving test

    const int n = 8*32*512;
    const int n_preview = 10;

    // Setup data
    SynchronisedArray<HalfData> halfdata = SynchronisedArray<HalfData>(context, n);
    for (int i=0; i<n; i++)
    {
        halfdata[i].a = i;
    }

    // Run kernel
    apply_kernel(context, queue, kernels["_half"], halfdata);

    // Preview results
    std::cout << "\n" << "Halving (first " << n_preview <<")\n";
    for(int i=0; i<n_preview; i++)
    {
        std::cout << halfdata[i].a << "/2 = "
                  << halfdata[i].b << "\n";
    }

    return 0;
}
