#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>
#include <cmath>

#define DIMENSION_MATRIZ 25

void multiplicar_matriz_vector(
    const std::vector<double> &A,
    const std::vector<double> &b,
    std::vector<double> &x,
    int rows,
    int cols)
{
    for (int i = 0; i < rows; i++)
    {
        double suma = 0.0;
        for (int j = 0; j < cols; j++)
        {
            suma += A[i * cols + j] * b[j];
        }
        x[i] = suma;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int filas_por_rank = std::ceil(DIMENSION_MATRIZ * 1.0 / nprocs);
    int total_rows = filas_por_rank * nprocs;
    int padding = total_rows - DIMENSION_MATRIZ;

    std::vector<double> A(total_rows * DIMENSION_MATRIZ, 0.0);
    std::vector<double> b(DIMENSION_MATRIZ, 1.0);
    std::vector<double> x(total_rows, 0.0);

    std::vector<double> A_local(filas_por_rank * DIMENSION_MATRIZ, 0.0);
    std::vector<double> x_local(filas_por_rank, 0.0);

    if (rank == 0)
    {
        for (int i = 0; i < DIMENSION_MATRIZ; i++)
        {
            for (int j = 0; j < DIMENSION_MATRIZ; j++)
            {
                A[i * DIMENSION_MATRIZ + j] = i + 1;
            }
        }

        fmt::print("DIMENSION_MATRIZ={}, nprocs={}, filas_por_rank={}, total_rows={}, padding={}\n",
                   DIMENSION_MATRIZ, nprocs, filas_por_rank, total_rows, padding);
    }

    MPI_Bcast(b.data(), DIMENSION_MATRIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(
        A.data(),
        filas_por_rank * DIMENSION_MATRIZ,
        MPI_DOUBLE,
        A_local.data(),
        filas_por_rank * DIMENSION_MATRIZ,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    // fmt::print("Matriz procesada por el proceso {}: \n", rank);
    // for (int i = 0; i < A_local.size(); i++)
    // {
    //     fmt::print("{:.2f} ", A_local[i]);
    // }
    // fmt::print("\n");
    int filas_validas = 0;
    int fila_inicial_global = rank * filas_por_rank;

    if (fila_inicial_global < DIMENSION_MATRIZ)
    {
        filas_validas = std::min(filas_por_rank, DIMENSION_MATRIZ - fila_inicial_global);
    }

    if (filas_validas > 0)
    {
        fmt::print("Proceso {}: filas_validas={}, fila_inicial_global={}\n", rank, filas_validas, fila_inicial_global);
        multiplicar_matriz_vector(A_local, b, x_local, filas_validas, DIMENSION_MATRIZ);
    }

    MPI_Gather(
        x_local.data(),
        filas_por_rank,
        MPI_DOUBLE,
        x.data(),
        filas_por_rank,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        fmt::print("Resultado real:\n");
        for (int i = 0; i < DIMENSION_MATRIZ; i++)
        {
            fmt::print("{:.2f} ", x[i]);
        }
        fmt::print("\n");
    }

    MPI_Finalize();
    return 0;
}