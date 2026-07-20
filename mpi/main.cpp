#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fmt/core.h>
#include <mpi.h>

std::vector<int> read_file()
{
    std::fstream fs("datos.txt", std::ios::in);
    std::string line;
    std::vector<int> ret;

    while (std::getline(fs, line))
    {
        if (!line.empty())
        {
            ret.push_back(std::stoi(line));
        }
    }

    fs.close();
    return ret;
}

void calcular_frecuencias(
    const std::vector<int> &datos_local,
    std::vector<int> &frecuencias_local,
    int elementos_validos)
{
    for (int i = 0; i < elementos_validos; i++)
    {
        int valor = datos_local[i];
        if (valor >= 0 && valor <= 100)
        {
            frecuencias_local[valor]++;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> datos;
    int total_elementos = 0;

    if (rank == 0)
    {
        datos = read_file();
        total_elementos = datos.size();
    }

    MPI_Bcast(&total_elementos, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_rank = std::ceil(total_elementos * 1.0 / nprocs);
    int total_rows = rows_per_rank * nprocs;
    int padding = total_rows - total_elementos;

    std::vector<int> A(total_rows, 0);
    std::vector<int> A_local(rows_per_rank, 0);

    std::vector<int> frecuencias(101, 0);
    std::vector<int> frecuencias_local(101, 0);

    long long suma_local = 0;
    long long suma_global = 0;

    int minimo_local = std::numeric_limits<int>::max();
    int maximo_local = std::numeric_limits<int>::min();

    int minimo_global = 0;
    int maximo_global = 0;

    if (rank == 0)
    {
        for (int i = 0; i < total_elementos; i++)
        {
            A[i] = datos[i];
        }

        fmt::print("total_elementos={}, nprocs={}, rows_per_rank={}, total_rows={}, padding={}\n",
                   total_elementos, nprocs, rows_per_rank, total_rows, padding);
    }

    MPI_Scatter(
        A.data(),
        rows_per_rank,
        MPI_INT,
        A_local.data(),
        rows_per_rank,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    int elementos_validos = 0;
    int indice_inicial_global = rank * rows_per_rank;

    if (indice_inicial_global < total_elementos)
    {
        elementos_validos = std::min(rows_per_rank, total_elementos - indice_inicial_global);
    }

    if (elementos_validos > 0)
    {
        for (int i = 0; i < elementos_validos; i++)
        {
            suma_local += A_local[i];
            minimo_local = std::min(minimo_local, A_local[i]);
            maximo_local = std::max(maximo_local, A_local[i]);
        }

        calcular_frecuencias(A_local, frecuencias_local, elementos_validos);
    }

    MPI_Reduce(
        &suma_local,
        &suma_global,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD);

    MPI_Reduce(
        &minimo_local,
        &minimo_global,
        1,
        MPI_INT,
        MPI_MIN,
        0,
        MPI_COMM_WORLD);

    MPI_Reduce(
        &maximo_local,
        &maximo_global,
        1,
        MPI_INT,
        MPI_MAX,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < 101; i++)
        {
            frecuencias[i] = frecuencias_local[i];
        }

        for (int src = 1; src < nprocs; src++)
        {
            std::vector<int> buffer_freq(101, 0);

            MPI_Recv(
                buffer_freq.data(),
                101,
                MPI_INT,
                src,
                1,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);

            for (int i = 0; i < 101; i++)
            {
                frecuencias[i] += buffer_freq[i];
            }
        }

        double promedio = 0.0;
        if (total_elementos > 0)
        {
            promedio = static_cast<double>(suma_global) / total_elementos;
        }

        fmt::print("+-------+--------+\n");
        fmt::print("| Valor | Conteo |\n");
        fmt::print("+-------+--------+\n");

        for (int i = 0; i <= 100; i++)
        {
            if (frecuencias[i] > 0)
            {
                fmt::print("| {:>5} | {:>6} |\n", i, frecuencias[i]);
            }
        }

        fmt::print("+-------+--------+\n");
        fmt::print("Promedio: {:.6f}\n", promedio);
        fmt::print("Minimo: {}\n", minimo_global);
        fmt::print("Maximo: {}\n", maximo_global);
    }
    else
    {
        MPI_Send(
            frecuencias_local.data(),
            101,
            MPI_INT,
            0,
            1,
            MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
