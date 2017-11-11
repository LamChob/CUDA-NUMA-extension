#include <numavec.h>
#include <iostream>
#include <chrono>
#include <numaif.h>

using namespace std;
using namespace chrono;

constexpr float N_IT { 100.0 };

void measure_access ( numavec<float>& c ) {
    auto t1 = system_clock::now(); 
    for ( int j = 0; j< N_IT; j++ ) {
        for ( int i = 0; i < c.size(); ++i ) {
            ++c[i];	
        }
    }
    auto t2 = system_clock::now();
     
    cout << c.size() << " " << c.size()*N_IT/duration_cast<microseconds>(t2-t1).count() << " ";
}

int main( int argc, char* argv[] ) {
    
    if ( argc < 3 ) exit(255);

    int SIZE = atoi(argv[1]);
    int core = atoi(argv[2]);

    unsigned long count {1};
    int flags {0};
    int* status = new int[count];
    float** pages = new float*[count];

    numavec<float> c(SIZE);

    measure_access(c);
    pages[0] = &c[0];
    move_pages(0, count, (void**) pages,NULL, status, flags);
    cout << " pos: " << status[0];

    c.migrate(core);
    
    pages[0] = &c[0];
    move_pages(0, count, (void**) pages,NULL, status, flags);
    cout << " pos: " << status[0] << " ";

    measure_access(c); 
    cout << endl;
}
