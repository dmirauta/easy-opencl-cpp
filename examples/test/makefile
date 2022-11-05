binary = test.out

test:
	CPLUS_INCLUDE_PATH="/opt/rocm/include" g++ -lOpenCL -std=c++23 -o ${binary} main.cpp

clean:
	rm ${binary}
	