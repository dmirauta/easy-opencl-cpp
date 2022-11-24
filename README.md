# easy-opencl-cpp

Naive attempt at an easyfied opencl c++ wrapper, tries to hide some of the boilerplate.

## Use

Files required:

* A header defining the structs that need to be streamed to and from gpu (e.g `datastructs.h`), this will be used on both sides.

* A cl file defining any kernels (e.g `kernels.cl`), each being expected to loop through an array of a datatype specified in `datastructs.h`.

    * To keep the code clean, the operations of a single work item could optionally be defined in one or more separate source files (e.g `kernelutils.c`). These could be C98 functions that can also be compiled and debugged ordinarily (with `gcc` for instance).

* Any source files used by the kernel.

A program (see `examples/test/main.cpp`) using this wrapper then:

* Includes `easy_cl.hpp`, `synchronised_array.hpp` and the user defined `datastructs.h`

* Creates an `EasyCL` object which initialises a device

* Loads the required kernels with `ecl.load_kernels(source_files, kernel_names, build_options)`

* Then for each loaded kernel it wishes to apply:

    * Creates one or more `SynchronisedArray` (SA)

        ```
        Dims dims(X, [Y, Z]) \\ array dimensions
        SynchronisedArray<DataStruct> data(ecl.context, [optional CL mem flags,] dims)
        ```

    * Write data to any arrays that need to be read from on the GPU (e.g `data[2,3].input_a = 2`, `data[2,3].input_b = 3.7`)

    * Applies one of the loaded kernels with at least one SA as input with `ecl.apply_kernel("fancy_addition", data_arr1, ...)`, the size of the first array is used to set the workgroup size

    * When the kernel is finished, the `SynchronisedArray` will update with any changes made by the kernel, for instance we might now have `data[2,3].output_c = 5.7`

## Compilation

See `examples/test/makefile`

## TODO

* Check for c++23 during compilation (and compile without multidimentional indexing otherwise?)

* GLCL interop?

* Need to account for different compilers potentially packing structs differently sometimes? (causing a mismatch in reading and writing between c++ and opencl?)
