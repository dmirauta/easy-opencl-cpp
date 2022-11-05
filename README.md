# easy-opencl-cpp

Naive attempt at easyfied opencl c++ interface, tries to hide some of the boilerplate.

## Use

Files required:

* A header defining the structs that need to be streamed to and from gpu (e.g `datastructs.h`), this will be used on both sides.

* A cl file defining any kernels (e.g `kernels.cl`), each being expected to loop through an array of a specified datatype.

    * To keep the code clean, the operations of a single iteration could optionally be defined in one or more separate source files (e.g `kernelutils.c`). These could be C98 functions that can also be compiled and debugged ordinarily (with `gcc` for instance).

* Any source files used by the kernel.

A program (see `examples/test`) using this wrapper then:

* Includes `src/easy_cl.hpp`

* Creates an `EasyCL` object which initialises a device

* Loads the required kernels with `ecl.load_kernels(source_files, kernel_names, build_options)`

* The for each loaded kernel it wishes to apply:

    * Creates a `SynchronisedArray<DataStruct> data(ecl.context, X [,Y ,Z] )`

    * Initilses this data (e.g `data[2,3].input_a = 2`, `data[2,3].input_b = 3.7`, `data[2,3].input_c = 0`)

    * Applies one of the loaded kernels on this object `apply_kernel(ecl, "_add", data)`

    * When the kernel is finished, the `SynchronisedArray` will update with any changes made by the kernel, for instance we might now have `data[2,3].input_c = 5.7`

## Compilation

May require pointing to OCL header location e.g.
`export CPLUS_INCLUDE_PATH="/opt/rocm/include"`

`g++ -lOpenCL -std=c++23 -o test.out main.cpp`

## TODO

* Check for c++23 during compilation (and compile without multidimentional indexing otherwise?)

* GLCL interop?

* unnecesarily copying some data to and from by packing like this, apply_kernel should, on top of normal/simple SynchronisedArray, have options along the lines of:

    * InputOnlySA (not copied back)

    * OutputOnlySA (not copied to, presumably instantiated on device)

    * A Param struct, to be shared (across all work items)

* also some concern for GPU packing structs differently?
