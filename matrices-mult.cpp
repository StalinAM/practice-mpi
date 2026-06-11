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

    if (rank == 0)
    {
        std::vector<double> A(MATRIX_DIM * MATRIX_DIM);
        std::vector<double> b(MATRIX_DIM);
        std::vector<double> x(MATRIX_DIM);

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
        int rows_per_rank = std::ceil(MATRIX_DIM * 1.0 / nprocs);
        int padding = rows_per_rank * nprocs - MATRIX_DIM;

        fmt::print("MATRIX_DIM: {}, nprocs: {}, rows_per_rank: {}, padding: {}\n",
                   MATRIX_DIM, nprocs, rows_per_rank, padding);

        // enviar dimensiones y datos
        for (int i = 1; i < nprocs; i++)
        {
            int filas = rows_per_rank;
            if (i == nprocs - 1)
            {
                filas = rows_per_rank - padding;
            }
            // enviar dimension
            std::vector<int> data = {MATRIX_DIM, filas};

            MPI_Send(
                data.data(),
                2, // data.size()
                MPI_INT,
                i,
                0,
                MPI_COMM_WORLD);

            const double *buffer = A.data();
            MPI_Send(
                &buffer[i * rows_per_rank * MATRIX_DIM],
                filas * MATRIX_DIM, // data.size()
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD);
            // enviar el vector b
            MPI_Send(
                b.data(),
                MATRIX_DIM, // data.size()
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD);
        }
        // fmt::print("RANK: {}, {} x {} \n", rank, rows_per_rank, MATRIX_DIM);
        // realizar multiplicacion
        multiplicar_matriz_vector(A, b, x, rows_per_rank, MATRIX_DIM);

        for (int i = 1; i < nprocs; i++)
        {
            int filas = rows_per_rank;
            if (i == nprocs - 1)
            {
                filas = rows_per_rank - padding;
            }

            MPI_Recv(
                x.data() + (i * rows_per_rank),
                filas, // data.size()
                MPI_DOUBLE,
                i,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        }
        // fmt::print("RANK: {}, resultado local x:\n", rank);
        imprimir_vector(x);
    }
    else
    {
        std::vector<double> b_local(MATRIX_DIM);

        std::vector<int> data_rec(2);
        MPI_Recv(
            data_rec.data(),
            2, // data.size()
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        int matrix_dim = data_rec[0];
        int rows = data_rec[1];

        // fmt::print("RANK: {}, {} x {} \n", rank, rows, matrix_dim);

        std::vector<double> A_local(rows * matrix_dim);

        MPI_Recv(
            A_local.data(),
            rows * matrix_dim, // data.size()
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        MPI_Recv(
            b_local.data(),
            MATRIX_DIM,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        if (rank == 2)
        {
            // imprimir_matriz(A_local, rows, matrix_dim);
            // fmt::print("\n");
            // imprimir_vector(b_local);
        }

        // realizar multiplicacion
        std::vector<double> x_local(rows);
        multiplicar_matriz_vector(A_local, b_local, x_local, rows, matrix_dim);

        // fmt::print("RANK: {}, resultado local x:\n", rank);
        // imprimir_vector(x_local);

        MPI_Send(
            x_local.data(),
            rows, // data.size()
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}