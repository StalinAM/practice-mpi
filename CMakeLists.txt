cmake_minimum_required(VERSION 3.16)

project(ejemplo-mpi)

find_package(fmt CONFIG REQUIRED)

set(MPI_ROOT "C:/Program Files (x86)/Intel/oneAPI/mpi/latest")
set(MPI_INCLUDE "${MPI_ROOT}/include")
set(MPI_LIB "${MPI_ROOT}/lib")

include_directories(
    ${MPI_INCLUDE}
    )
add_executable(ejemplo-mpi main.cpp)

target_link_libraries(ejemplo-mpi PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )

add_executable(matrices-mult matrices-mult.cpp)

target_link_libraries(matrices-mult PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )
add_executable(matrices-mult-no matrices-mult-no.cpp)

target_link_libraries(matrices-mult-no PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )

add_executable(busqueda busqueda.cpp)

target_link_libraries(busqueda PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )
add_executable(escala-grises escala-grises.cpp)

target_link_libraries(escala-grises PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )
add_executable(escala-grises-no escala-grises-no.cpp)

target_link_libraries(escala-grises-no PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )

add_executable(testeo testeo.cpp)

target_link_libraries(testeo PRIVATE 
    fmt::fmt
    ${MPI_LIB}/impi.lib
   )

