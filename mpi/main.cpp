#include <iostream>
#include <fmt/core.h>

#include <mpi.h>

int main(int argc, char **argv)
{
    // inicializar MPI
    MPI_Init(&argc, &argv);
    // sacar numero de procesos
    int nprocs;
    int rank;

    // ranks

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // version de MPI

    int version, subversion;
    MPI_Get_version(&version, &subversion);

    if (rank == 0)
    {
        fmt::print("MPI Version: {}.{}\n", version, subversion);
        fmt::print("Number of processes: {}\n", nprocs);
    }

    fmt::print("RANK_{} de procesos/{}\n", rank, nprocs);

    std::cout.flush();

    /*
    while(1<2){

    }
    */

    MPI_Finalize();

    return 0;
}