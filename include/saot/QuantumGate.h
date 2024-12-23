#ifndef SAOT_QUANTUM_GATE_H
#define SAOT_QUANTUM_GATE_H

#include "saot/UnitaryPermMatrix.h"
#include "saot/Polynomial.h"
#include "saot/ScalarKind.h"
#include "utils/square_matrix.h"

#include <array>
#include <variant>

namespace saot {

enum GateKind : int {
  gUndef = 0,

  // single-qubit gates
  gX = -10,
  gY = -11,
  gZ = -12,
  gH = -13,
  gP = -14,
  gRX = -15,
  gRY = -16,
  gRZ = -17,
  gU = -18,

  // two-qubit gates
  gCX = -100,
  gCNOT = -100,
  gCZ = -101,
  gSWAP = -102,
  gCP = -103,

  // general multi-qubit dense gates
  gU1q = 1,
  gU2q = 2,
  gU3q = 3,
  gU4q = 4,
  gU5q = 5,
  gU6q = 6,
  gU7q = 7,
  // to be defined by nqubits directly
};

GateKind String2GateKind(const std::string& s);
std::string GateKind2String(GateKind t);

inline std::ostream& printConstantMatrix(
    std::ostream& os,
    const utils::square_matrix<std::complex<double>>& cMat) {
  return utils::printComplexMatrixF64(os, cMat);
}

std::ostream& printParametrizedMatrix(
    std::ostream& os,
    const utils::square_matrix<saot::Polynomial>& cMat);

class GateMatrix {
public:
  // specify gate matrix with (up to three) parameters
  using gate_params_t =
      std::array<std::variant<std::monostate, int, double>, 3>;
  // unitary permutation matrix type
  using up_matrix_t = saot::UnitaryPermutationMatrix;
  // constant matrix type
  using c_matrix_t = utils::square_matrix<std::complex<double>>;
  // parametrised matrix type
  using p_matrix_t = utils::square_matrix<saot::Polynomial>;
  // signature matrix type
  using sig_matrix_t = utils::square_matrix<std::complex<ScalarKind>>;

private:
  enum ConvertibleKind : int {
    Unknown = -1,
    Convertible = 1,
    UnConvertible = 0
  };
  struct Cache {
    ConvertibleKind isConvertibleToUpMat;
    up_matrix_t upMat;
    ConvertibleKind isConvertibleToCMat;
    c_matrix_t cMat;
    // gate matrix is always convertible to pMat
    ConvertibleKind isConvertibleToPMat;
    p_matrix_t pMat;

    double sigZeroTol;
    double sigOneTol;
    sig_matrix_t sigMat;

    Cache()
      : isConvertibleToUpMat(Unknown), upMat(), isConvertibleToCMat(Unknown),
        cMat(), isConvertibleToPMat(Unknown), pMat(), sigZeroTol(-1.0),
        sigOneTol(-1.0), sigMat() {}
  };

  mutable Cache cache;

  void computeAndCacheUpMat(double tolerance) const;
  void computeAndCacheCMat() const;
  void computeAndCachePMat() const;
  void computeAndCacheSigMat(double zeroTol, double oneTol) const;

public:
  GateKind gateKind;
  gate_params_t gateParameters;

  GateMatrix() : cache(), gateKind(gUndef), gateParameters() {}

  GateMatrix(GateKind gateKind, const gate_params_t &params = {})
      : cache(), gateKind(gateKind), gateParameters(params) {}

  GateMatrix(const up_matrix_t& upMat);
  GateMatrix(const c_matrix_t& upMat);
  GateMatrix(const p_matrix_t& upMat);

  static GateMatrix FromName(const std::string& name,
                             const gate_params_t &params = {});

  void permuteSelf(const llvm::SmallVector<int>& flags);

  // get cached unitary perm matrix object associated with this GateMatrix
  const up_matrix_t* getUnitaryPermMatrix(double tolerance = 0.0) const {
    if (cache.isConvertibleToUpMat == Unknown)
      computeAndCacheUpMat(tolerance);
    if (cache.isConvertibleToUpMat == UnConvertible)
      return nullptr;
    return &cache.upMat;
  }

  up_matrix_t* getUnitaryPermMatrix(double tolerance = 0.0) {
    if (cache.isConvertibleToUpMat == Unknown)
      computeAndCacheUpMat(tolerance);
    if (cache.isConvertibleToUpMat == UnConvertible)
      return nullptr;
    return &cache.upMat;
  }

  // get cached constant matrix object associated with this GateMatrix
  const c_matrix_t* getConstantMatrix() const {
    if (cache.isConvertibleToCMat == Unknown)
      computeAndCacheCMat();
    if (cache.isConvertibleToCMat == UnConvertible)
      return nullptr;
    return &cache.cMat;
  }

  c_matrix_t* getConstantMatrix() {
    if (cache.isConvertibleToCMat == Unknown)
      computeAndCacheCMat();
    if (cache.isConvertibleToCMat == UnConvertible)
      return nullptr;
    return &cache.cMat;
  }

  // get cached parametrized matrix object associated with this GateMatrix
  const p_matrix_t& getParametrizedMatrix() const {
    if (cache.isConvertibleToPMat == Unknown)
      computeAndCachePMat();
    assert(cache.isConvertibleToPMat == Convertible);
    return cache.pMat;
  }

  p_matrix_t& getParametrizedMatrix() {
    if (cache.isConvertibleToPMat == Unknown)
      computeAndCachePMat();
    assert(cache.isConvertibleToPMat == Convertible);
    return cache.pMat;
  }

  const sig_matrix_t& getSignatureMatrix(double zeroTol, double oneTol) const {
    if (cache.sigMat.edgeSize() == 0) {
      assert(cache.sigZeroTol < 0.0);
      assert(cache.sigOneTol < 0.0);
      computeAndCacheSigMat(zeroTol, oneTol);
    }
    return cache.sigMat;
  }

  bool isConvertibleToUnitaryPermMatrix(double tolerance) const {
    return getUnitaryPermMatrix(tolerance) != nullptr;
  }

  bool isConvertibleToConstantMatrix() const {
    return getConstantMatrix() != nullptr;
  }

  // @brief Get number of qubits
  int nqubits() const;

  std::ostream& printCMat(std::ostream& os) const {
    const auto* cMat = getConstantMatrix();
    assert(cMat);
    return saot::printConstantMatrix(os, *cMat);
  }

  std::ostream& printPMat(std::ostream& os) const {
    return saot::printParametrizedMatrix(os, getParametrizedMatrix());
  }

  // preset unitary permutation matrices
  static const up_matrix_t MatrixI1_up;
  static const up_matrix_t MatrixI2_up;

  static const up_matrix_t MatrixX_up;
  static const up_matrix_t MatrixY_up;
  static const up_matrix_t MatrixZ_up;

  static const up_matrix_t MatrixCX_up;
  static const up_matrix_t MatrixCZ_up;

  // preset constant matrices
  static const c_matrix_t MatrixI1_c;
  static const c_matrix_t MatrixI2_c;

  static const c_matrix_t MatrixX_c;
  static const c_matrix_t MatrixY_c;
  static const c_matrix_t MatrixZ_c;
  static const c_matrix_t MatrixH_c;

  static const c_matrix_t MatrixCX_c;
  static const c_matrix_t MatrixCZ_c;

  // preset parametrized matrices
  static const p_matrix_t MatrixI1_p;
  static const p_matrix_t MatrixI2_p;

  static const p_matrix_t MatrixX_p;
  static const p_matrix_t MatrixY_p;
  static const p_matrix_t MatrixZ_p;
  static const p_matrix_t MatrixH_p;

  static const p_matrix_t MatrixCX_p;
  static const p_matrix_t MatrixCZ_p;
};

class QuantumGate {
private:
  mutable int opCountCache = -1;

public:
  /// The canonical form of qubits is in ascending order
 llvm::SmallVector<int> qubits;
  GateMatrix gateMatrix;

  QuantumGate() : qubits(), gateMatrix() {}

  QuantumGate(const GateMatrix& gateMatrix, int q)
      : qubits({q}), gateMatrix(gateMatrix) {
    assert(gateMatrix.nqubits() == 1);
  }

  QuantumGate(GateMatrix&& gateMatrix, int q)
    : qubits({q}), gateMatrix(gateMatrix) {
    assert(gateMatrix.nqubits() == 1);
  }

  QuantumGate(const GateMatrix& gateMatrix, std::initializer_list<int> qubits)
      : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nqubits() == qubits.size());
    sortQubits();
  }

  QuantumGate(GateMatrix&& gateMatrix, std::initializer_list<int> qubits)
    : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nqubits() == qubits.size());
    sortQubits();
  }

  QuantumGate(const GateMatrix& gateMatrix, const llvm::SmallVector<int>& qubits)
    : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nqubits() == qubits.size());
    sortQubits();
  }

  QuantumGate(GateMatrix&& gateMatrix, const llvm::SmallVector<int>& qubits)
    : qubits(qubits), gateMatrix(gateMatrix) {
    assert(gateMatrix.nqubits() == qubits.size());
    sortQubits();
  }

  int nqubits() const { return qubits.size(); }

  bool isQubitsSorted() const {
    if (qubits.empty())
      return true;
    for (unsigned i = 0; i < qubits.size() - 1; i++) {
      if (qubits[i + 1] <= qubits[i])
        return false;
    }
    return true;
  }

  bool checkConsistency() const {
    return (gateMatrix.nqubits() == qubits.size());
  }

  std::ostream& displayInfo(std::ostream& os) const;

  int findQubit(int q) const {
    for (unsigned i = 0; i < qubits.size(); i++) {
      if (qubits[i] == q)
        return i;
    }
    return -1;
  }

  void sortQubits();

  /// @brief B.lmatmul(A) will return AB. That is, gate B will be applied first.
  QuantumGate lmatmul(const QuantumGate& other) const;

  int opCount(double zeroSkippingThres = 1e-8) const;

  bool isConvertibleToUnitaryPermGate(double tolerance) const {
    return gateMatrix.isConvertibleToUnitaryPermMatrix(tolerance);
  }

  bool isConvertibleToConstantGate() const {
    return gateMatrix.isConvertibleToConstantMatrix();
  }

  static QuantumGate I1(int q) {
    return QuantumGate(GateMatrix::MatrixI1_c, q);
  }

  static QuantumGate I2(int q0, int q1) {
    return QuantumGate(GateMatrix::MatrixI2_c, {q0, q1});
  }

  static QuantumGate H(int q) {
    return QuantumGate(GateMatrix::MatrixH_c, q);
  }

  static QuantumGate RandomU1q(int q) {
    return QuantumGate(GateMatrix(utils::randomUnitaryMatrix(2)), q);
  }

  static QuantumGate RandomU2q(int q0, int q1) {
    return QuantumGate(GateMatrix(utils::randomUnitaryMatrix(4)), {q0, q1});
  }

  static QuantumGate RandomU3q(int q0, int q1, int q2) {
    return QuantumGate(
      GateMatrix(utils::randomUnitaryMatrix(8)), {q0, q1, q2});
  }
};

} // namespace saot

#endif // SAOT_QUANTUM_GATE_H