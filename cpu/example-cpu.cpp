#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// #define print

using namespace std;

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

void compute_cpu(const float *a, const float *b, float *c, unsigned m, unsigned n, unsigned p)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < p; col++)
        {
            auto sum = 0.0;
            // Compute result of ONE element of C
            for (int i = 0; i < n; i++)
                sum += a[row * n + i] * b[i * p + col];
            c[row * p + col] = sum;
        }
    }
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

    print_matrix(a, m, n);
    print_matrix(b, n, p);
    print_matrix(c, m, p);

    auto start = std::chrono::steady_clock::now();
    compute_cpu(a, b, c, m, n, p);
    cout << "[CPU]\t\tElapsed(ms)=" << since(start).count() << "\n";

    print_matrix(c, m, p);

    if (validate_result(c, m, p, (float)(n + 1) / 2 * n))
    {
        cout << "[RESULT]\tCORRECT"
             << "\n";
    }
    else
    {
        cout << "[RESULT]\tWRONG"
             << "\n";
    }

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}