binary = test.out

test:
	CPLUS_INCLUDE_PATH="/opt/rocm/include" g++ -lOpenCL -std=c++23 -o ${binary} tests.cpp

clean:
	rm ${binary}
	