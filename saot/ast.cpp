#include "saot/ast.h"
#include "saot/CircuitGraph.h"

#include "utils/iocolor.h"
#include "utils/utils.h"

using namespace IOColor;
using namespace saot;
using namespace saot::ast;

namespace {
std::ostream& printGateMatrix(
    std::ostream& os, const GateMatrix::gate_params_t &params) {
  if (std::get_if<std::monostate>(&params[0]))
    return os;
  if (const int* v = std::get_if<int>(&params[0]))
    os << "%" <<* v;
  else if (const double* v = std::get_if<double>(&params[0]))
    os << std::setprecision(8) << std::scientific << *v;

  if (std::get_if<std::monostate>(&params[1]))
    return os;
  if (const int* v = std::get_if<int>(&params[1]))
    os << ",%" <<* v;
  else if (const double* v = std::get_if<double>(&params[1]))
    os << "," << std::setprecision(8) << std::scientific << *v;

  if (std::get_if<std::monostate>(&params[2]))
    return os;
  if (const int* v = std::get_if<int>(&params[2]))
    os << ",%" <<* v;
  else if (const double* v = std::get_if<double>(&params[2]))
    os << "," << std::setprecision(8) << std::scientific << *v;

  return os;
}
} // anonymous namespace

std::ostream& MeasureStmt::print(std::ostream& os) const {
  os << "m ";
  utils::printArrayNoBrackets(os, std::span(qubits), ' ');
  return os << "\n";
}

std::ostream& QuantumCircuit::print(std::ostream& os) const {
  os << "circuit<nqubits=" << nqubits
     << ", nparams=" << nparams << "> " << name << " {\n";
  for (const auto& s : stmts)
    s->print(os);

  if (!paramDefs.empty()) {
    os << "\n";
    for (const auto& def : paramDefs)
      def.print(os);
  }

  return os << "}\n";
}

std::ostream& GateApplyStmt::print(std::ostream& os) const {
  os << name;
  // parameter
  std::visit([&os]<typename Arg_t>(Arg_t&& arg) {
      using T = std::decay_t<Arg_t>;
      if constexpr (std::is_same_v<T, int>) {
        os << "(#" << arg << ")";
        return;
      }
      if constexpr (std::is_same_v<T, GateMatrix::gate_params_t>) {
        printGateMatrix(os << "(", arg) << ")";
        return;
      }
    }, paramRefOrMatrix);

  // target qubits
  os << " ";
  utils::printArrayNoBrackets(os, std::span(qubits), ' ');
  return os;
}

std::ostream& GateChainStmt::print(std::ostream& os) const {
  const auto size = gates.size();
  if (size == 0)
    return os;
  os << "  ";
  for (size_t i = 0; i < size - 1; i++)
    gates[i].print(os) << "@ ";
  gates[size - 1].print(os) << ";\n";
  return os;
}

std::ostream& ParameterDefStmt::print(std::ostream& os) const {
  assert(false && "Not Implemented");
  return os;
  // os << "#" << refNumber << " = { ";
  // if (gateMatrix.isConstantMatrix()) {
  //     auto it = gateMatrix.cData().cbegin();
  //     utils::print_complex(os, *it);
  //     while (++it != gateMatrix.cData().cend())
  //         utils::print_complex(os << ", ", *it);
  //     return os << " }\n";
  // }

  // assert(gateMatrix.isParametrizedMatrix());
  // auto it = gateMatrix.pData().cbegin();
  // it->print(os);
  // while (++it != gateMatrix.pData().cend())
  //     it->print(os << ", ");
  // return os << " }\n";
}

QuantumGate QuantumCircuit::gateApplyToQuantumGate(
    const GateApplyStmt& gaStmt) const {
  if (const auto* p = std::get_if<int>(&gaStmt.paramRefOrMatrix)) {
    const auto v = *p;
    auto it = std::ranges::find_if(std::as_const(paramDefs),
      [v](const ParameterDefStmt &stmt) {
        return stmt.refNumber == v;
      });
    if (it != paramDefs.cend())
      return QuantumGate(*it->gateMatrix, gaStmt.qubits);
    assert(false && "Cannot find parameter def stmt");
    return QuantumGate();
  }
  if (const auto* p =
          std::get_if<GateMatrix::gate_params_t>(&gaStmt.paramRefOrMatrix))
    return QuantumGate(GateMatrix::FromName(gaStmt.name, *p), gaStmt.qubits);
  return QuantumGate(GateMatrix::FromName(gaStmt.name), gaStmt.qubits);
}

CircuitGraph QuantumCircuit::toCircuitGraph() const {
  CircuitGraph graph;
  for (const auto& s : stmts) {
    if (s->isNot(Statement::SK_GateChain))
      continue;
    const auto* chain = static_cast<const GateChainStmt*>(s.get());
    if (chain == nullptr) {
      std::cerr << BOLDYELLOW("Warning: ")
                << "Unable to convert to GateChainStmt when calling "
                   "RootNode::toCircuitGraph\n";
      continue;
    }
    if (chain->gates.empty())
      continue;

    auto quGate = gateApplyToQuantumGate(chain->gates[0]);
    for (int i = 1; i < chain->gates.size(); i++)
      quGate = quGate.lmatmul(gateApplyToQuantumGate(chain->gates[i]));
    graph.addGate(quGate);
  }
  return graph;
}

QuantumCircuit QuantumCircuit::FromCircuitGraph(const CircuitGraph& G) {
  const auto allBlocks = G.getAllBlocks();

  QuantumCircuit qc;
  int paramRefNumber = 0;

  for (const auto* B : allBlocks) {
    const auto qubits = B->getQubits();
    std::string gateName = "u" + std::to_string(qubits.size()) + "q";
    qc.stmts.push_back(
        std::make_unique<GateApplyStmt>(gateName, paramRefNumber, qubits));

    qc.paramDefs.emplace_back(
      paramRefNumber, std::make_unique<GateMatrix>(B->quantumGate->gateMatrix));
    paramRefNumber++;
  }

  return qc;
}