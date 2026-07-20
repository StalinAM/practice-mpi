cmake_minimum_required(VERSION 3.16)

project(Ejemplo)

enable_language(CXX CUDA)

find_package(fmt CONFIG REQUIRED)
find_package(SFML COMPONENTS System Graphics Window CONFIG REQUIRED)

add_executable(ejemplo 
    main.cpp
    kernel.cu
)


target_include_directories(ejemplo PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(ejemplo PRIVATE fmt::fmt
    SFML::System SFML::Graphics SFML::Window)