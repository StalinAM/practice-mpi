#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>
#include <cmath>

#define DIMENSION_VECTOR 25
#define NUM_SEARCH 10

int search_in_vector(const std::vector<double> &vec, double value)
{
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] == value)
        {
            return i;
        }
    }

    return -1;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int datos_por_rank = std::ceil(DIMENSION_VECTOR * 1.0 / nprocs);
    int total_datos = datos_por_rank * nprocs;
    int padding = total_datos - DIMENSION_VECTOR;

    std::vector<double> A({1, 2, 3, 4, 5, 6, 7, 8, 9, 12,
                           11, 12, 13, 14, 15, 16, 10, 10, 19, 20,
                           21, 22, 10, 24, 25});
    std::vector<double> A_local(datos_por_rank, 0.0);

    std::vector<int> search_results(nprocs, -1);
    int result_local = -1;
    if (rank == 0)
    {
        fmt::print("DIMENSION_VECTOR={}, nprocs={}, datos_por_rank={}, total_datos={}, padding={}\n",
                   DIMENSION_VECTOR, nprocs, datos_por_rank, total_datos, padding);
    }

    MPI_Scatter(
        A.data(),
        datos_por_rank,
        MPI_DOUBLE,
        A_local.data(),
        datos_por_rank,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    int filas_validas = 0;
    int fila_inicial_global = rank * datos_por_rank;

    if (fila_inicial_global < DIMENSION_VECTOR)
    {
        filas_validas = std::min(datos_por_rank, DIMENSION_VECTOR - fila_inicial_global);
    }

    if (filas_validas > 0)
    {
        int index = search_in_vector(A_local, 10.0);
        result_local = index;
    }

    MPI_Gather(
        &result_local,
        1,
        MPI_INT,
        search_results.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        fmt::print("Resultado real:\n");
        for (int i = 0; i < nprocs; i++)
        {
            fmt::print("{} ", search_results[i]);
        }
        fmt::print("\n");
    }

    MPI_Finalize();
    return 0;
}