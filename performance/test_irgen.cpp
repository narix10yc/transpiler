// #include "gen_file.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

typedef struct { double data[8]; } v8double;

extern "C" {
void u3_f64_0_00003fff(double*, double*, uint64_t, uint64_t, v8double);
void u3_f64_1_01003fff(double*, double*, uint64_t, uint64_t, v8double);
void u3_f64_2_02003fff(double*, double*, uint64_t, uint64_t, v8double);
void u3_f64_3_03003fff(double*, double*, uint64_t, uint64_t, v8double);
}

void simulate_circuit(double* real, double* imag, uint64_t idxMax) {
//   u3_f64_0_00003fff(real, imag, 0, 1,
    // (v8double){0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554});
//   u3_f64_1_01003fff(real, imag, 0, 1,
    // (v8double){0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554});
//   u3_f64_2_02003fff(real, imag, 0, 1,
    // (v8double){0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554});
  u3_f64_3_03003fff(real, imag, 0, idxMax,
    (v8double){0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554});
}


using real_ty = double;
size_t k = 3;
size_t s = 3;

std::string getCurrentTime() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d-%H%M%S");
    return ss.str();
}

std::string getOutputFilename() {
    return "out_" + getCurrentTime() + ".csv";
}

#include "timeit/timeit.h"
#include <functional>

using namespace timeit;

int main() {
    Timer timer;
    TimingResult tr;
    real_ty *real, *imag;
    uint64_t nqubits;

    auto setup = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
            imag = (real_ty*) aligned_alloc(64, sizeof(real_ty) * (1<<nqubits));
        };
    };

    auto teardown = [&]() {
        free(real); free(imag);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    };

    // open output file
    auto filename = getOutputFilename();
    std::ofstream f { filename };
    if (!f.is_open()) {
        std::cerr << "failed to open " << filename << "\n";
        return 1;
    }

    f << "method,compiler,test_name,real_ty,num_threads,nqubits,k,s,t_min,t_q1,t_med,t_q3\n";
    f << std::scientific;

    for (uint64_t nqubit = 4; nqubit < 30; nqubit += 2) {
        tr = timer.timeit(
            [&](){ simulate_circuit(real, imag, (1<<(nqubit-3))); },
            setup(nqubit),
            teardown
        );
        std::cerr << "nqubits = " << nqubit << "\n";
        tr.display();
        f << "ir_gen" // method
          << "," << "clang-17" // compiler
          << "," << "u3" // test_name
          << "," << "f" << 8*sizeof(real_ty) // real_ty
          << "," << 1 // num_threads
          << "," << nqubit // nqubits
          << "," << k // k
          << "," << s // s
          << ","
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}