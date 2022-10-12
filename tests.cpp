#include <iostream>

#include "quick_cl.hpp"             // this includes cpp files, so only test.cpp file needs to be compiled
//#include "synchronised_array.hpp" // imports are set up like this so that needed versions of templated classes are automatically picked up
#include "datastructs.c"

using namespace std;

int main(int argc, char* argv[]) {

    //cout << "Still working on line " << __LINE__ << "!\n";

    bool verbose;
    if (argc > 1 && strcmp(argv[1], "1") == 0)
        verbose = true;
    else
        verbose = false;

    cl::Context context;
    cl::Device device;
    cl::CommandQueue queue;
    setup_cl(context, device, queue, verbose);

    string build_options = "-D HALVE_IS_QUARTER";
    vector<string> source_files{"datastructs.c", "tests.cl"};
    vector<string> kernel_names{"_add", "_halve"};
    map<string, cl::Kernel> kernels = setup_cl_prog(context, device, source_files, kernel_names, build_options, verbose);

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
    cout << "\n" << "Adding (viewing last "<<_m_preview<<"x"<<_m_preview<<")\n";
    for(int i=m1-_m_preview; i<m1; i++)
    {
        for(int j=m2-_m_preview; j<m2; j++)
        {
            cout << adddata[i, j].a << " + "
                 << adddata[i, j].b << " = "
                 << adddata[i, j].c << "\n";
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Halving test

    const int n = 8*32*512;
    const int n_preview = 10;

    // Setup data
    SynchronisedArray<HalveData> halfdata = SynchronisedArray<HalveData>(context, n);
    for (int i=0; i<n; i++)
    {
        halfdata[i].a = i;
    }

    // Run kernel
    apply_kernel(context, queue, kernels["_halve"], halfdata);

    // Preview results
    cout << "\n" << "Halving (viewing first " << n_preview <<")\n";
    for(int i=0; i<n_preview; i++)
    {
        cout << halfdata[i].a << "/2 = "
             << halfdata[i].b << "\n";
    }

    return 0;
}
