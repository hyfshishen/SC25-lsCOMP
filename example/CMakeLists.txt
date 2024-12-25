# Find CUDA package
find_package(CUDA REQUIRED)

set(install_dir ${PROJECT_BINARY_DIR})
set(cuLSZ_cssi "cuLSZ_cssi")
set(cuLSZ_xpcs "cuLSZ_xpcs")
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# Add include and library directories
include_directories(${INCLUDE_DIR})

# Compile headers as a library
cuda_add_library(cuLSZ_libs STATIC ${SRC_DIR}/cuLSZ_kernel.cu
                                   ${SRC_DIR}/cuLSZ_utility.cu
                                   ${SRC_DIR}/cuLSZ_timer.cu
                                   ${SRC_DIR}/cuLSZ_entry.cu)

# Compile executable binary
cuda_add_executable(${cuLSZ_cssi} cuLSZ_cssi.cpp)
cuda_add_executable(${cuLSZ_xpcs} cuLSZ_xpcs.cpp)

# Link with headers
target_link_libraries(${cuLSZ_cssi} cuLSZ_libs)
target_link_libraries(${cuLSZ_xpcs} cuLSZ_libs)

# Set output paths for the compiled binary
set_target_properties(${cuLSZ_cssi} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cuLSZ_xpcs} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary.
install(TARGETS ${cuLSZ_cssi} DESTINATION bin)
install(TARGETS ${cuLSZ_xpcs} DESTINATION bin)