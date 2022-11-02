# easy-opencl-cpp

Naive attempt at easyfied opencl c++ interface, tries to hide some of the boilerplate.

## Compilation

May require pointing to OCL header location e.g.
`export CPLUS_INCLUDE_PATH="/opt/rocm/include"`

`g++ -lOpenCL -std=c++23 -o test.out tests.cpp`

## TODO

* Check for c++23 during compilation

* Makefile

* GLCL interop?

* unnecesarily copying some data to and from by packing like this, apply_kernel should, on top of normal SynchronisedArray have options along the lines of:

    * InputOnlySA (not copied back)

    * OutputOnlySA (not copied to, presumably instantiated on device)

    * A Param struct, to be shared (across all work items)

* also some concern for GPU packing structs differently?
