#include <iostream>

#include "easy_cl.hpp" 
#include "synchronised_array.hpp" // header only, templated class
#include "datastructs.h"          // struct definitions used in both c++ and opencl

using namespace std;

int main(int argc, char* argv[]) {

    //cout << "Still working on line " << __LINE__ << "!\n";

    bool verbose;
    if (argc > 1 && strcmp(argv[1], "1") == 0)
        verbose = true;
    else
        verbose = false;

    EasyCL ecl(verbose);

    vector<string> source_files{"datastructs.h", "kernelutils.c", "kernels.cl"};
    vector<string> kernel_names{"mult_add", "halve_or_quarter"};
    ecl.load_kernels(source_files, kernel_names, "-D DO_QUARTER");


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Halve or quorter test

    const int n = 8*32*512;
    const int n_preview = 10;

    // Setup data
    Dims d2(n);
    SynchronisedArray<HoQData> hoqdata(ecl.context, d2);
    for (int i=0; i<n; i++)
    {
        hoqdata[i].in = i;
    }

    // Run kernel
    ecl.apply_kernel("halve_or_quarter", hoqdata);

    // Preview results
    cout << "\n" << "Halving or quartering (depending on build options) (viewing first " << n_preview <<")\n";
    for(int i=0; i<n_preview; i++)
    {
        cout << hoqdata[i].in  << " -> "
             << hoqdata[i].out << "\n";
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Adding test

    const int m1 = 1024;
    const int m2 = 768;
    const int _m_preview = 3;

    // Setup data
    Dims d(m1, m2);
    SynchronisedArray<MultAddInData> madatain(ecl.context, CL_MEM_WRITE_ONLY, d);
    SynchronisedArray<MultAddOutData> madataout(ecl.context, CL_MEM_READ_ONLY, d);

    for (int i=0; i<m1; i++)
    {
        for (int j=0; j<m2; j++)
        {
            //cout << i << ", " << j << "\n"; 
            madatain[i, j].in1 = i;
            madatain[i, j].in2 = j;
        }
    }

    // Run kernel
    ecl.apply_kernel("mult_add", madatain, madataout);
    
    // Preview results
    cout << "\n" << "Mult and Add kernel results (viewing first " << n_preview <<")\n";
    for(int i=m1-_m_preview; i<m1; i++)
    {
        for(int j=m2-_m_preview; j<m2; j++)
        {
            cout << madatain[i, j].in1 << " + "
                 << madatain[i, j].in2 << " = "
                 << madataout[i, j].add << "    ";

            cout << madatain[i, j].in1 << " * "
                 << madatain[i, j].in2 << " = "
                 << madataout[i, j].mult << "\n";
        }
    }


    return 0;
}
