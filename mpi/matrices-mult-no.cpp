#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

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
    std::vector<double> b_local(DIMENSION_MATRIZ, 1.0);
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

        // El proceso 0 envía a cada proceso su bloque de filas de A
        for (int i = 1; i < nprocs; i++)
        {
            int index_inicial = i * filas_por_rank * DIMENSION_MATRIZ;

            MPI_Send(
                A.data() + index_inicial,
                filas_por_rank * DIMENSION_MATRIZ,
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD);
            MPI_Send(
                b.data(),
                DIMENSION_MATRIZ,
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD);
        }
    }
    else
    {
        // Cada proceso distinto de 0 recibe su bloque de filas
        MPI_Recv(
            A_local.data(),
            filas_por_rank * DIMENSION_MATRIZ,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(
            b_local.data(),
            DIMENSION_MATRIZ,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        int filas_validas = 0;
        int fila_inicial_global = rank * filas_por_rank;

        if (fila_inicial_global < DIMENSION_MATRIZ)
        {
            filas_validas = std::min(filas_por_rank, DIMENSION_MATRIZ - fila_inicial_global);
        }

        if (filas_validas > 0)
        {
            fmt::print("Proceso {}: filas_validas={}, fila_inicial_global={}\n",
                       rank, filas_validas, fila_inicial_global);

            multiplicar_matriz_vector(A_local, b_local, x_local, filas_validas, DIMENSION_MATRIZ);
        }
    }

    if (rank == 0)
    {
        multiplicar_matriz_vector(A, b, x, filas_por_rank, DIMENSION_MATRIZ);

        // El proceso 0 recibe los resultados parciales de los demás
        for (int i = 1; i < nprocs; i++)
        {
            int index_inicial = i * filas_por_rank;

            MPI_Recv(
                x.data() + index_inicial,
                filas_por_rank,
                MPI_DOUBLE,
                i,
                1,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        }

        fmt::print("Resultado real:\n");
        for (int i = 0; i < DIMENSION_MATRIZ; i++)
        {
            fmt::print("{:.2f} ", x[i]);
        }
        fmt::print("\n");
    }
    else
    {
        // Cada proceso distinto de 0 envía su resultado parcial al proceso 0
        MPI_Send(
            x_local.data(),
            filas_por_rank,
            MPI_DOUBLE,
            0,
            1,
            MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}