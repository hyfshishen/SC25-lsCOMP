# Find CUDA package
find_package(CUDA REQUIRED)

set(install_dir ${PROJECT_BINARY_DIR})
set(xpcs_conversion "xpcs_conversion")
set(cssi_conversion "cssi_conversion")
set(xpcs_truncation "xpcs_truncation")
set(cssi_truncation "cssi_truncation")
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
cuda_add_executable(${xpcs_conversion} xpcs_conversion.cpp)
cuda_add_executable(${cssi_conversion} cssi_conversion.cpp)
cuda_add_executable(${xpcs_truncation} xpcs_truncation.cpp)
cuda_add_executable(${cssi_truncation} cssi_truncation.cpp)

# Link with headers
target_link_libraries(${xpcs_conversion} cuLSZ_libs)
target_link_libraries(${cssi_conversion} cuLSZ_libs)
target_link_libraries(${xpcs_truncation} cuLSZ_libs)
target_link_libraries(${cssi_truncation} cuLSZ_libs)

# Set output paths for the compiled binary
set_target_properties(${xpcs_conversion} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cssi_conversion} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${xpcs_truncation} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})
set_target_properties(${cssi_truncation} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${install_dir})

# Set installation paths for the compiled binary.
install(TARGETS ${xpcs_conversion} DESTINATION bin)
install(TARGETS ${cssi_conversion} DESTINATION bin)
install(TARGETS ${xpcs_truncation} DESTINATION bin)
install(TARGETS ${cssi_truncation} DESTINATION bin)