binary = test.out
src_path = $(shell echo $$(dirname $$(dirname $$PWD)))/src
include_path = CPLUS_INCLUDE_PATH="/opt/rocm/include:${src_path}"

test:
	${include_path} g++ -lOpenCL -std=c++23 -o ${binary} ${src_path}/easy_cl.cpp main.cpp -g

clean:
	-rm core*
	-rm ${binary}
	