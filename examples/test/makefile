binary = test.out
src_path = $(shell echo $$(dirname $$(dirname $$PWD)))/src
set_include_path = export CPLUS_INCLUDE_PATH="/opt/rocm/include:${src_path}"

test:
	${set_include_path}
	g++ -lOpenCL -std=c++23 -o ${binary} ${src_path}/easy_cl.cpp main.cpp

clean:
	rm ${binary}
	