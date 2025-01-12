#include "saot/Parser.h"
#include "saot/CircuitGraph.h"
#include "saot/Fusion.h"
#include "openqasm/parser.h"

using namespace saot;

int main(int argc, const char** argv) {
  openqasm::Parser qasmParser("../examples/rqc/q6_67_52.qasm", 0);
  auto qasmRoot = qasmParser.parse();
  // Parser parser("../examples/rqc/q6_67_52.qasm");
  // auto qc = parser.parseQuantumCircuit();
  // qc.print(std::cerr);

  CircuitGraph graph;
  // qc.toCircuitGraph(graph);
  qasmRoot->toCircuitGraph(graph);
  graph.print(std::cerr << "Before Fusion:\n", 2) << "\n";


  CPUFusionConfig config = CPUFusionConfig::Default;
  NaiveCostModel costModel(3, 0, 1e-8);

  applyCPUGateFusion(config, &costModel, graph);
  graph.print(std::cerr << "After Fusion:\n", 2) << "\n";

  auto fusedQC = ast::QuantumCircuit::FromCircuitGraph(graph);
  fusedQC.print(std::cerr << "Fused QuantumCircuit Serialization\n") << "\n";


  return 0;
}