#include <iostream>
#include <fmt/core.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Convierte una porción de imagen RGBA a gris.
// Procesa solo 'filas' filas válidas.
void convertir_porcion_a_gris(
    const std::vector<uint8_t> &A_local,
    std::vector<uint8_t> &x_local,
    int filas,
    int columnas)
{
    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            int index = (i * columnas + j) * 4;

            uint8_t r = A_local[index];
            uint8_t g = A_local[index + 1];
            uint8_t b = A_local[index + 2];

            double valor_gris = 0.21 * r + 0.72 * g + 0.07 * b;
            uint8_t gris = static_cast<uint8_t>(valor_gris);

            x_local[index] = gris;
            x_local[index + 1] = gris;
            x_local[index + 2] = gris;
            x_local[index + 3] = A_local[index + 3]; // conserva alpha
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ancho = 0;
    int alto = 0;
    int canales = 0;

    uint8_t *pixeles_disco = nullptr;

    if (rank == 0)
    {
        pixeles_disco = stbi_load("original.jpg", &ancho, &alto, &canales, STBI_rgb_alpha);
        canales = 4;

        if (!pixeles_disco)
        {
            fmt::print("Error: no se pudo cargar la imagen.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&ancho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alto, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_rank = std::ceil(alto * 1.0 / nprocs);
    int total_rows = rows_per_rank * nprocs;
    int padding = total_rows - alto;

    // Imagen global con padding
    std::vector<uint8_t> A(total_rows * ancho * 4, 0);
    std::vector<uint8_t> x(total_rows * ancho * 4, 0);

    // Bloques locales
    std::vector<uint8_t> A_local(rows_per_rank * ancho * 4, 0);
    std::vector<uint8_t> x_local(rows_per_rank * ancho * 4, 0);

    if (rank == 0)
    {
        int total_bytes_reales = ancho * alto * 4;
        for (int i = 0; i < total_bytes_reales; i++)
        {
            A[i] = pixeles_disco[i];
        }

        fmt::print(
            "alto={}, ancho={}, nprocs={}, rows_per_rank={}, total_rows={}, padding={}\n",
            alto, ancho, nprocs, rows_per_rank, total_rows, padding);
    }

    MPI_Scatter(
        A.data(),
        rows_per_rank * ancho * 4,
        MPI_UINT8_T,
        A_local.data(),
        rows_per_rank * ancho * 4,
        MPI_UINT8_T,
        0,
        MPI_COMM_WORLD);

    int filas_validas = 0;
    int fila_inicial_global = rank * rows_per_rank;

    if (fila_inicial_global < alto)
    {
        filas_validas = std::min(rows_per_rank, alto - fila_inicial_global);
    }

    if (filas_validas > 0)
    {
        fmt::print("Proceso {}: filas_validas={}, fila_inicial_global={}\n",
                   rank, filas_validas, fila_inicial_global);

        convertir_porcion_a_gris(A_local, x_local, filas_validas, ancho);
    }

    MPI_Gather(
        x_local.data(),
        rows_per_rank * ancho * 4,
        MPI_UINT8_T,
        x.data(),
        rows_per_rank * ancho * 4,
        MPI_UINT8_T,
        0,
        MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::vector<uint8_t> imagen_final_gris(ancho * alto);

        for (int i = 0; i < ancho * alto; i++)
        {
            imagen_final_gris[i] = x[i * 4];
        }

        stbi_write_png("img-gris.png", ancho, alto, 1, imagen_final_gris.data(), ancho);

        stbi_image_free(pixeles_disco);

        fmt::print("Imagen en gris guardada como img-gris.png\n");
    }

    MPI_Finalize();
    return 0;
}