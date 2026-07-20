#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

void multiplicarMatriz(
    const std::vector<double> &A,
    const std::vector<double> &b,
    std::vector<double> &x,
    int ancho,
    int alto)
{
    for (int i = 0; i < alto; i++)
    {
        double suma = 0.0;
        for (int j = 0; j < ancho; j++)
        {
            int index = i * ancho + j;
            suma += A[index] * b[j];
        }
        x[i] = suma;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int num_procesos;
    int proceso;

    int ancho = 25;
    int alto = 10;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procesos);
    MPI_Comm_rank(MPI_COMM_WORLD, &proceso);

    int filas_por_rank = std::ceil(alto * 1.0 / num_procesos);
    int total_filas = filas_por_rank * num_procesos;
    int agregadas = total_filas - alto;

    std::vector<double> A(total_filas * ancho, 0.0);
    std::vector<double> A_local(filas_por_rank * ancho, 0.0);
    std::vector<double> b(ancho, 1.0);
    std::vector<double> x(total_filas, 0.0);
    std::vector<double> x_local(filas_por_rank, 0.0);

    if (proceso == 0)
    {
        for (int i = 0; i < alto; i++)
        {
            for (int j = 0; j < ancho; j++)
            {
                int index = i * ancho + j;
                A[index] = i + 1;
            }
        }

        fmt::print("ancho={}, alto={}, num_procesos={}, filas_por_rank={}, total_filas={}, agregadas={}\n",
                   ancho, alto, num_procesos, filas_por_rank, total_filas, agregadas);

        for (int destino = 1; destino < num_procesos; destino++)
        {
            MPI_Send(&ancho, 1, MPI_INT, destino, 0, MPI_COMM_WORLD);
            MPI_Send(&alto, 1, MPI_INT, destino, 0, MPI_COMM_WORLD);

            int index_inicial = destino * filas_por_rank * ancho;

            MPI_Send(
                A.data() + index_inicial,
                filas_por_rank * ancho,
                MPI_DOUBLE,
                destino,
                0,
                MPI_COMM_WORLD);

            MPI_Send(
                b.data(),
                ancho,
                MPI_DOUBLE,
                destino,
                0,
                MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&ancho, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&alto, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(
            A_local.data(),
            filas_por_rank * ancho,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        MPI_Recv(
            b.data(),
            ancho,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
    }

    int filas_validas = 0;
    int fila_inicial = proceso * filas_por_rank;

    if (fila_inicial < alto)
    {
        filas_validas = std::min(filas_por_rank, alto - fila_inicial);
    }

    if (filas_validas > 0)
    {
        multiplicarMatriz(A_local, b, x_local, ancho, filas_validas);
    }

    if (proceso == 0)
    {
        multiplicarMatriz(A, b, x, ancho, filas_por_rank);
        for (int i = 1; i < num_procesos; i++)
        {
            int index = i * filas_por_rank;

            MPI_Recv(
                x.data() + index,
                filas_por_rank,
                MPI_DOUBLE,
                i,
                1,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        }

        fmt::print("proceso {} resultado:\n", proceso);
        for (int i = 0; i < alto; i++)
        {
            fmt::print("{:.2f} ", x[i]);
        }
        fmt::print("\n");
    }
    else
    {
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