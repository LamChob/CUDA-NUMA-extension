cmake_minimum_required(VERSION 3.0)

find_package(CUDA REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set (CUDA_SOURCES src/tests/test-numa-interface.cu) 
set (BINARY test-numa-interface) 

set (BW_SOURCES src/tests/h2hbw.cu) 
set (BW_BINARY h2hbw)

set (AL_SOURCES src/tests/alloc.cu) 
set (AL_BINARY alloc)

set (APP_SOURCES src/tests/app-benchmark.cu) 
set (APP_BINARY app-benchmark)

set (INC_DIRS "src/migration/" "src/interface" "src/tests" "inc/") 
 
link_libraries(-lpthread -lhwloc -lnuma)
include_directories(${BINARY} PUBLIC ${INC_DIRS}) 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 --expt-extended-lambda -gencode arch=compute_61,code=compute_61  -DUSE_PAGE_MIGRATION -O3 -Xcompiler -fopenmp)

#cuda_add_executable(${BINARY} ${CUDA_SOURCES}) 
cuda_add_executable(${BW_BINARY} ${BW_SOURCES}) 
cuda_add_executable(${AL_BINARY} ${AL_SOURCES}) 
cuda_add_executable(${APP_BINARY} ${APP_SOURCES}) 
