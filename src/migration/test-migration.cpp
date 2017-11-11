#include <migration.h>
#include <iostream>
#include <chrono>
#include <numaif.h>

using namespace std;
using namespace chrono;

void print_pos(float* d) {
    unsigned long count = 1;
    int flags {0};
    int* status = new int[count];
    float** pages = new float*[count];

    pages[0] = d;
    move_pages(0, count, (void**) pages,NULL, status, flags);
    cout << " pos: " << status[0] << endl;

    delete[] status;
    delete[] pages;
}

int main(int argc, char* argv[]) {
    //assert(!(argc < 2));
    char *s;

    int _size = atoi(argv[1]);

    float* _data = new float[_size];
    for(int i = 0; i < _size; i++ ) _data[i] = 1.0f;
    migrate((void**) &_data, _size*sizeof(float), 0);
    cout << "Original Pointer: " << _data << endl;
    cout << "Current Position: "; print_pos(_data);

    auto t1  = system_clock::now();
    migrate((void**) &_data, _size*sizeof(float), 1);
    auto t2  = system_clock::now();
    double mt = (double) duration_cast<nanoseconds>(t2-t1).count();
    cout << "New Pointer: " << _data << endl;
    cout << "New Position: "; print_pos(_data);
    cout << "Bandwidth: " << (_size * sizeof(float)) / mt << " GByte/s" << endl;

    delete _data;
}

