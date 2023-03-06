binary = test.out
header_path = $(shell echo $$(dirname $$(dirname $$PWD)))
include_path = CPLUS_INCLUDE_PATH="/opt/rocm-5.2.3/include:${header_path}"

test:
	${include_path} g++ -lOpenCL -std=c++23 -o ${binary} main.cpp -g

clean:
	-rm core*
	-rm ${binary}
	