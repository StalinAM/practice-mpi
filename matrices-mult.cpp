#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>

#define MATRIX_DIM 25

void imprimir_matriz(const std::vector<double> &A, int rows, int cols)
{
    // fmt::print("Matriz A local:\n");
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fmt::print("{:.2f} ", A[i * cols + j]);
        }
        fmt::print("\n");
    }
}
void imprimir_vector(const std::vector<double> &b)
{
    // fmt::print("Vector b local:\n");
    for (size_t i = 0; i < b.size(); i++)
    {
        fmt::print("{:.2f} ", b[i]);
    }
    fmt::print("\n");
}
void multiplicar_matriz_vector(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &x, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        double suma = 0;
        for (int j = 0; j < cols; j++)
        {
            int index = i * cols + j;
            suma += A[index] * b[j];
        }
        x[i] = suma;
    }
}
int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int nprocs;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_per_rank = std::ceil(MATRIX_DIM * 1.0 / nprocs);
    int total_rows = rows_per_rank * nprocs;
    int padding = total_rows - MATRIX_DIM;

    std::vector<double> A(total_rows * MATRIX_DIM);
    std::vector<double> b(MATRIX_DIM);
    std::vector<double> x(MATRIX_DIM);

    std::vector<double> A_local(rows_per_rank * MATRIX_DIM);
    std::vector<double> x_local(rows_per_rank);

    if (rank == 0)
    {

        // inicializar la matriz A y el vector b

        for (int i = 0; i < MATRIX_DIM; i++)
        {
            for (int j = 0; j < MATRIX_DIM; j++)
            {
                int index = i * MATRIX_DIM + j;
                A[index] = i;
            }
        }

        for (int i = 0; i < MATRIX_DIM; i++)
        {
            b[i] = 1;
        }

        // nmero de filas para cada RANK (proceso)
        fmt::print("MATRIX_DIM: {}, nprocs: {}, rows_per_rank: {}, total_rows: {}, padding: {}\n",
                   MATRIX_DIM, nprocs, rows_per_rank, total_rows, padding);
    }

    MPI_Bcast(
        b.data(),
        MATRIX_DIM,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    MPI_Scatter(
        A.data(),
        rows_per_rank * MATRIX_DIM,
        MPI_DOUBLE,

        A_local.data(),
        rows_per_rank * MATRIX_DIM,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD);

    multiplicar_matriz_vector(
        A_local,
        b,
        x_local,
        rows_per_rank,
        MATRIX_DIM);

    MPI_Gather(
        x_local.data(),
        rows_per_rank,
        MPI_DOUBLE,

        x.data(),
        rows_per_rank,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        imprimir_vector(x);
    }

    MPI_Finalize();

    return 0;
}