#include <sycl/sycl.hpp>
#include <iostream>

using namespace std;
using namespace sycl;

void print_matrix(const float *matrix, const int m, const int n)
{
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

void compute_gpu(const float *a, const float *b, const float *c, unsigned m, unsigned n, unsigned p)
{
    queue q(default_selector_v);
    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    {
        buffer<float, 1> a_buf(a, range(m * n));
        buffer<float, 1> b_buf(b, range(n * p));
        buffer<float, 1> c_buf(c, range(m * p));

        q.submit([&](auto &h)
                {
            // Read from a and b, write to c
            accessor a(a_buf, read_only);
            accessor b(b_buf, read_only);
            accessor c(c_buf, sycl::write_only, sycl::no_init);
            sycl::stream out(1024, 256, h);

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
                out << sum << "\n";
            }); 
        });
        q.wait();
    }
}

int main()
{
    constexpr int m = 2, n = 4, p = 6;

    float *a = new float[m * n];
    float *b = new float[n * p];
    float *c = new float[m * p];

    fill_matrix(a, m, n, [](unsigned row, unsigned col){return 1.0f;});
    fill_matrix(b, n, p, [](unsigned row, unsigned col){return (float)(row*n+col);});
    fill_matrix(c, m, p, [](unsigned row, unsigned col){return 0.0f;});

    print_matrix(a, m, n);
    print_matrix(b, n, p);
    print_matrix(c, m, p);

    compute_gpu(a, b, c, m, n, p);

    print_matrix(c, m, p);
    return 0;
}