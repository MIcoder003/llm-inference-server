cmake_minimum_required(VERSION 3.14)
project(my_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force llama.cpp to build its common utils library even though
# we're not building it as the top-level/standalone project
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

add_subdirectory(externals/llama.cpp)

add_executable(llm-inference-server src/main.cpp)

target_include_directories(llm-inference-server PRIVATE
    ${CMAKE_SOURCE_DIR}/include 
    externals/llama.cpp/common
)

target_link_libraries(llm-inference-server PRIVATE llama-common llama)