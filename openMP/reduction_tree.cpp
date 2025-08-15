
/*implementation of a parallel sum using reduction tree */ 

#include <iostream>
#include <omp.h> 
#include <vector>
#include <cmath> 

constexpr int  N = 8; 

int main (int argc, char *argv[]) {

    std::vector<int> x(N); 
    
    for(int i =0; i < N; ++i)
        x[i] = i + 1; 

    std::vector<int> s = x; 

    int num_levels = std::log2(N); 

    std::cout << "Initial array:"; 
    for (auto v : s)
        std::cout << v << " ";
    std::cout << "\n\n"; 

    //reduction tree 

    for(int level = 0; level < num_levels; level++)
    {
        int num_pairs = N >> (level + 1);

        #pragma omp parallel for
        for(int i = 0; i < num_pairs; i++)
        {
            int idx1 = i * 2; 
            int idx2 = i * 2 + 1; 
            s[i] = s[idx1] + s[idx2]; 

            int tid = omp_get_thread_num(); 
            #pragma omp critical 
            std::cout << "level " << level + 1 << ", Thread " << tid
                << " adds s[" << idx1 << "] + s [" << idx2 << "] = " << s[i] << "\n"; 
        }

        std::cout << "Partial sums after level " << level + 1 << ": ";

        for (int i = 0; i < (N >> (level + 1)); i++) std::cout << s[i] << " ";

        std::cout << "\n\n";
    }
 
   std::cout << "Total sum: " << s[0] << "\n";
    
    return 0;
}
