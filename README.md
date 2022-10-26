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
