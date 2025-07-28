#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

using namespace std;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ double delt_d;
__device__ double freeze_temp_d;

__device__ double initial_d;
__device__ double bound_d;

__device__ int problem_size_d;
__device__ double dx_d;
__device__ double dt_d;

__device__ int get_pos(int i, int j, int k)
{
    return i + j * problem_size_d + k * problem_size_d * problem_size_d;
}

__device__ double c_ro(double t)
{
    double c_frosen = 1600;
    double ro_frosen = 1200;

    double c_melt = 1900;
    double ro_melt = 1200;

    double L = 330000 * 0.1;

    if (t < freeze_temp_d - delt_d)
        return c_frosen * ro_frosen;

    if (t >= freeze_temp_d - delt_d && t < freeze_temp_d)
        return (c_frosen + L / delt_d / 2) * ro_frosen;

    if (t >= freeze_temp_d && t < freeze_temp_d + delt_d)
        return (c_melt + L / delt_d / 2) * ro_melt;

    if (t >= freeze_temp_d + delt_d)
        return c_melt * ro_melt;

    return 0;
}

__device__ double k(double t)
{
    double k_frosen = 0.92;

    double k_melt = 0.72;

    double scale = 3600;

    if (t < freeze_temp_d)
        return k_frosen * scale;

    if (t >= freeze_temp_d)
        return k_melt * scale;

    return 0;
}

__global__ void init(double *in)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx < problem_size_d && idy < problem_size_d && idz < problem_size_d) {
        if (idx > 0 && idx < problem_size_d - 1 && idy > 0 && idy < problem_size_d - 1 && idz > 0 &&
            idz < problem_size_d - 1) {
            in[get_pos(idx, idy, idz)] = initial_d;
        } else {
            in[get_pos(idx, idy, idz)] = bound_d;
        }
    }
}

__global__ void solve(double *out, double *in)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx > 0 && idx < problem_size_d - 1 && idy > 0 && idy < problem_size_d - 1 && idz > 0 &&
        idz < problem_size_d - 1)
        out[get_pos(idx, idy, idz)] =
            in[get_pos(idx, idy, idz)] +
            dt_d / ((dx_d * dx_d) * c_ro(in[get_pos(idx, idy, idz)])) *
                ((((k(in[get_pos(idx + 1, idy, idz)]) + k(in[get_pos(idx, idy, idz)])) / 2.0) *
                      (in[get_pos(idx + 1, idy, idz)] - in[get_pos(idx, idy, idz)]) -
                  ((k(in[get_pos(idx, idy, idz)]) + k(in[get_pos(idx - 1, idy, idz)])) / 2.0) *
                      (in[get_pos(idx, idy, idz)] - in[get_pos(idx - 1, idy, idz)])) +
                 (((k(in[get_pos(idx, idy + 1, idz)]) + k(in[get_pos(idx, idy, idz)])) / 2.0) *
                      (in[get_pos(idx, idy + 1, idz)] - in[get_pos(idx, idy, idz)]) -
                  ((k(in[get_pos(idx, idy, idz)]) + k(in[get_pos(idx, idy - 1, idz)])) / 2.0) *
                      (in[get_pos(idx, idy, idz)] - in[get_pos(idx, idy - 1, idz)])) +
                 (((k(in[get_pos(idx, idy, idz + 1)]) + k(in[get_pos(idx, idy, idz)])) / 2.0) *
                      (in[get_pos(idx, idy, idz + 1)] - in[get_pos(idx, idy, idz)]) -
                  ((k(in[get_pos(idx, idy, idz)]) + k(in[get_pos(idx, idy, idz - 1)])) / 2.0) *
                      (in[get_pos(idx, idy, idz)] - in[get_pos(idx, idy, idz - 1)])));
}

int main()
{
    int max_size = 300;

    double dt = 0.0001;
    double dx = 0.01;
    double delt = 0.1;

    double initial = 15;
    double freeze_temp = 0;

    gpuErrchk(cudaMemcpyToSymbol(dt_d, &dt, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(dx_d, &dx, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(initial_d, &initial, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(freeze_temp_d, &freeze_temp, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(delt_d, &delt, sizeof(double)));

    double *heat_array_old;
    gpuErrchk(
        cudaMalloc((void **)&heat_array_old, max_size * max_size * max_size * sizeof(double)));

    double *heat_array_now;
    gpuErrchk(
        cudaMalloc((void **)&heat_array_now, max_size * max_size * max_size * sizeof(double)));

    double *heat_array_cpu = (double *)malloc(max_size * max_size * max_size * sizeof(double));

    ofstream file_out("out.txt");

    for (int j = -10; j <= -10; j += 5) {
        for (int i = 40; i <= 40; i += 5) {
            int problem_size = i * 0.01 / dx + 1;
            gpuErrchk(cudaMemcpyToSymbol(problem_size_d, &problem_size, sizeof(int)));

            double bound_val = j;
            gpuErrchk(cudaMemcpyToSymbol(bound_d, &bound_val, sizeof(double)));

            dim3 threadsPerBlock(8, 8, 8);
            dim3 numBlocks(problem_size / threadsPerBlock.x + 1,
                           problem_size / threadsPerBlock.y + 1,
                           problem_size / threadsPerBlock.z + 1);

            file_out << "Air temperature:" << bound_val << "C "
                     << "Briquette side size:" << i * 0.01 << "m" << endl;
            file_out << "Time(hours)\tMaximum temperature of the briquette(C)" << endl;

            init<<<numBlocks, threadsPerBlock>>>(heat_array_old);
            init<<<numBlocks, threadsPerBlock>>>(heat_array_now);

            double centre = 0;

            for (int time = 0; time <= 8 / dt; time++) {
                solve<<<numBlocks, threadsPerBlock>>>(heat_array_now, heat_array_old);

                double *tmp = heat_array_now;
                heat_array_now = heat_array_old;
                heat_array_old = tmp;

                if (time % ((int)(1 / dt)) == 0) {
                    gpuErrchk(cudaMemcpy(
                        &centre,
                        &heat_array_old[problem_size / 2 + problem_size / 2 * problem_size +
                                        problem_size / 2 * problem_size * problem_size],
                        sizeof(double), cudaMemcpyDeviceToHost));
                    file_out << time * dt << "\t" << centre << endl;

                    gpuErrchk(
                        cudaMemcpy(heat_array_cpu, heat_array_old,
                                   problem_size * problem_size * problem_size * sizeof(double),
                                   cudaMemcpyDeviceToHost));
                    char out_string[100];
                    sprintf(out_string, "plot/result_%d.vtk", time);
                    ofstream out(out_string);
                    out << "# vtk DataFile Version 2.0" << endl;
                    out << "Heat" << endl;
                    out << "ASCII" << endl;
                    out << "DATASET STRUCTURED_POINTS" << endl;
                    out << "DIMENSIONS " << problem_size << " " << problem_size << " "
                        << problem_size << endl;
                    out << "ASPECT_RATIO 1 1 1" << endl;
                    out << "ORIGIN 0 0 0" << endl;
                    out << "POINT_DATA " << problem_size * problem_size * problem_size << endl;
                    out << "SCALARS heat float 1" << endl;
                    out << "LOOKUP_TABLE default" << endl;
                    for (int i = 0; i < problem_size; i++) {
                        for (int j = 0; j < problem_size; j++)
                            for (int k = 0; k < problem_size; k++) {
                                out << heat_array_cpu[i + j * problem_size +
                                                      k * problem_size * problem_size]
                                    << " ";
                            }
                        out << endl;
                    }
                    out.close();
                }
            }

            file_out << endl;
        }
    }

    return 0;
}
