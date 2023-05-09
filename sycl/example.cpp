#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

// #define print

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
    q.wait();
}

bool validate_result(const float *a, unsigned m, unsigned n, float value)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (abs(a[row * n + col] - value) > 0.001f)
            {
                cout << "[ERROR]\t\ta[" << row << "][" << col << "]==" << a[row * n + col] << " != " << value << "\n";
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
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

    fill_matrix(a, m, n, [](unsigned row, unsigned col)
                { return 1.0f; });
    fill_matrix(b, n, p, [](unsigned row, unsigned col)
                { return (float)(row + 1); });
    fill_matrix(c, m, p, [](unsigned row, unsigned col)
                { return 0.0f; });

    queue q(gpu_selector_v);
    cout << "[Device]\t" << q.get_device().get_info<info::device::name>() << "\n";

    print_matrix(a, m, n);
    print_matrix(b, n, p);
    print_matrix(c, m, p);

    auto start = std::chrono::steady_clock::now();
    compute_gpu(q, a, b, c, m, n, p);
    cout << "[GPU]\t\tElapsed(ms)=" << since(start).count() << "\n";

    print_matrix(c, m, p);

    if (validate_result(c, m, p, (float)(n+1)/2 * n)) {
        cout << "[RESULT]\tCORRECT" << "\n";
    } else {
        cout << "[RESULT]\tWRONG" << "\n";
    }

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}