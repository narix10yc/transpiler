// #include "gen_file.h"

#include <string>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

typedef double v8double __attribute__((vector_size(64)));

extern "C" {
void u3_0_02003fff(double*, double*, uint64_t, uint64_t, v8double);
void u3_1_02001080(double*, double*, uint64_t, uint64_t, v8double);
void u3_2_02003fc0(double*, double*, uint64_t, uint64_t, v8double);
}

void simulate_circuit(double* real, double* imag, size_t idxMax) {
  u3_0_02003fff(real, imag, 0, idxMax,
    (v8double){0.7455743899704769,-0.6611393443073451,0.5681414468755827,0.6795542237022189,0,0.08374721744037222,0.3483304829644841,0.3067364145782554});
  u3_1_02001080(real, imag, 0, idxMax,
    (v8double){1,0,0,-1,0,0,0,0});
  u3_2_02003fc0(real, imag, 0, idxMax,
    (v8double){0.9984574954665696,-0.05552143501950479,0.05552143501950479,0.9984574954665696,0,0,0,0});
}


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
    double *real, *imag;
    uint64_t nqubits;

    auto setup = [&](uint64_t _nqubits) {
        nqubits = _nqubits;
        return [&]() {
            real = (double*) malloc(sizeof(double) * (1<<nqubits));
            imag = (double*) malloc(sizeof(double) * (1<<nqubits));
        };
    };

    auto teardown = [&]() {
        free(real); free(imag);
        std::this_thread::sleep_for(std::chrono::seconds(3));
    };

    // open output file
    auto filename = getOutputFilename();
    std::ofstream f { filename };
    if (!f.is_open()) {
        std::cerr << "failed to open " << filename << "\n";
        return 1;
    }

    f << "method,test_name,real_ty,nqubits,k,s,t_min,t_q1,t_med,t_q3\n";
    f << std::scientific;

    for (uint64_t nqubit = 4; nqubit < 28; nqubit += 2) {
        tr = timer.timeit(
            [&](){ simulate_circuit(real, imag, (1<<(nqubit-3))); },
            setup(nqubit),
            teardown
        );
        std::cerr << "nqubits = " << nqubit << "\n";
        tr.display();
        f << "template,u3,f64," << nqubit << ",2,2,"
          << tr.min << "," << tr.q1 << "," << tr.med << "," << tr.q3 << "\n";
    }
    
    return 0;
}