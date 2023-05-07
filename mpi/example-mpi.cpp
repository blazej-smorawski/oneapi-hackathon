#include <sycl/sycl.hpp>
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

//#define print

using namespace std;
using namespace sycl;

template <
    class result_t = std::chrono::milliseconds,
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const &start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

void print_matrix(const float *matrix, const int m, const int n)
{
#ifdef print
    cout << "[";
    for (int row = 0; row < m; row++)
    {
        cout << "[";
        for (int col = 0; col < n; col++)
        {
            cout << matrix[row * n + col];
            if (col != n - 1)
            {
                cout << ',';
            }
        }
        cout << "]";
        if (row != m - 1)
        {
            cout << "\n";
        }
    }
    cout << "]"
         << "\n\n";
#endif
}

void fill_matrix(float *matrix, const int m, const int n, float (*fn)(unsigned row, unsigned col))
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            matrix[row * n + col] = fn(row, col);
        }
    }
}

void compute_gpu(queue &q, const float *a, const float *b, float *c, unsigned m, unsigned n, unsigned p)
{
    buffer<float> a_buf(a, m * n);
    buffer<float> b_buf(b, n * p);
    buffer<float> c_buf(c, m * p);

    q.submit([&](auto &h)
             {
        // Read from a and b, write to c
        accessor a(a_buf, h, read_only);
        accessor b(b_buf, h, read_only);
        accessor c(c_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(range(m, p), [=](auto index) 
        {
            // Threading index that iterates over C.
            int row = index[0];
            int col = index[1];
            auto sum = 0.0;
            // Compute result of ONE element of C
            for (int i = 0; i < n; i++)
                sum += a[row * n + i] * b[i * p + col];
            c[row * p + col] = sum;
        }); });
}

void compute_cpu(const float *a, const float *b, float *c, unsigned m, unsigned n, unsigned p)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            auto sum = 0.0;
            // Compute result of ONE element of C
            for (int i = 0; i < n; i++)
                sum += a[row * n + i] * b[i * p + col];
            c[row * p + col] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 4)
    {
        cout << "USAGE: ./example M N P"
             << "\n";
        return 1;
    }

    int m = atoi(argv[1]), n = atoi(argv[2]), p = atoi(argv[3]);

    float *a = new float[m * n];
    float *b = new float[n * p];
    float *c = new float[m * p];
    int offset = 0;

    queue q(gpu_selector_v);
    cout << "[Device]\t" << q.get_device().get_info<info::device::name>() << "\n";

    if(world_rank == 0) {
        fill_matrix(a, m, n, [](unsigned row, unsigned col)
                { return 1.0f; });
        fill_matrix(b, n, p, [](unsigned row, unsigned col)
                    { return (float)(row + 1); });
        fill_matrix(c, m, p, [](unsigned row, unsigned col)
                    { return 0.0f; });
    }

    auto start = std::chrono::steady_clock::now();

    if(world_rank == 0) {
        MPI_Send(a, m*n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(b, n*p, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else {
        offset = (m/2) * n;
        MPI_Recv(a, m*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, n*p, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    print_matrix(a, m, n);
    print_matrix(b, n, p);
    print_matrix(c, m, p);

    compute_gpu(q, a + offset, b, c + offset, m/2, n, p);
    print_matrix(c, m, p);

    if(world_rank == 0) {
        int other_offset = (m/2) * n;
        MPI_Recv(c + other_offset, m/2 * p, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Send(c + offset, m/2 * p, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    cout << "[GPU]\t\tElapsed(ms)=" << since(start).count() << "\n";

    cout<< "[FINAL]\t\t" << "\n";
    print_matrix(c, m, p);

    delete[] a;
    delete[] b;
    delete[] c;

    MPI_Finalize();

    return 0;
}