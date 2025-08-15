
#include <ranges> 
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>
#include <algorithm> 
#include <execution> 

void print_vector(const std::vector<double>& vec)
{
    for (int val : vec) 
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void initialize(std::vector<double> &vec_x, std::vector<double> &vec_y)
{
    assert(vec_x.size() == vec_y.size()); 

    auto ints = std::views::iota(0); 

    std::for_each_n(std::execution::par_unseq, ints.begin(), vec_x.size(), [&vec_x](int i){vec_x[i] = (double)i;});

    std::fill_n(std::execution::par_unseq, vec_y.begin(), vec_y.size(), 2.); 
}


void daxpy_kernel(const double a, const std::vector<double>& vec_x, std::vector<double>& vec_y)
{
    std::size_t n = vec_x.size();
    assert(vec_y.size() == n);

    __m256d a_vec = _mm256_set1_pd(a);
    std::size_t i = 0;

    for (; i + 3 < n; i += 4)
    {
        __m256d x_vec = _mm256_loadu_pd(&vec_x[i]); 
        __m256d y_vec = _mm256_loadu_pd(&vec_y[i]);
        __m256d result = _mm256_fmadd_pd(a_vec, x_vec, y_vec);
        _mm256_storeu_pd(&vec_y[i], result);
    }

    for (; i < n; ++i)
    {
        vec_y[i] += a * vec_x[i];
    }
}

bool check(double a, std::vector<double> const &vec_y); 

int main (int argc, char *argv[]) 
{
    if(argc != 2)
    {
        std::cerr << "Error: missing argument!" << std::endl; 
        return 1; 
    }

    long long n = std::stoll(argv[1]); 

    std::vector<double> vec_x(n, 0); 
    std::vector<double> vec_y(n, 0);

    double a = 2.0; 

    initialize(vec_x, vec_y); 

    daxpy(a, vec_x, vec_y); 

    if(!check(a, vec_y))
    {
        std::cerr << "Error!" << std::endl; 
        return 1; 
    }

    std::cerr << "OK!" << std::endl;

    /*mesaure bandwidth */ 
    using clt_t = std::chrono::steady_clock; 
    daxpy(a, vec_x, vec_y); 

    auto start = clt_t::now(); 

    int nit = 100; 

    for (int it = 0; it < nit; ++it)
    {
        daxpy(a, vec_x, vec_y); 
    }

    auto seconds = std::chrono::duration<double>(clt_t::now() - start).count(); 
    auto gigabytes = 3. * static_cast<double>(vec_x.size()) * 
        static_cast<double>(sizeof(double)) * 
        static_cast<double>(nit) * 1.e-9; 

    std::cerr << "Bandwidth [GB/S]: " << (gigabytes/seconds) << std::endl; 

    return 0; 
}

bool check(double a, std::vector<double> const &vec_y)
{
    double tolerance = 2. * std::numeric_limits<double>::epsilon(); 

    for(std::size_t x = 0; x < vec_y.size(); ++x)
    {
        double should = a * x + 2;
        if(std::abs(vec_y[x] - should) > tolerance)
        {
            return false; 
        }
    }
    return true; 
}
