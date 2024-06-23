#ifndef QUENCH_GATE_MATRIX_H
#define QUENCH_GATE_MATRIX_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>

namespace quench::cas {

class Polynomial;

class CASNode {
public:
    struct expr_value {
        bool isConstant;
        double value;
    };

    virtual std::ostream& print(std::ostream&) const = 0;

    virtual std::ostream& printLaTeX(std::ostream&) const = 0;

    virtual expr_value getExprValue() const = 0;

    virtual bool equals(const CASNode*) const = 0;

    bool equals(std::shared_ptr<CASNode> p) const {
        return equals(p.get());
    }

    virtual int getSortPriority() const = 0;

    virtual Polynomial toPolynomial() const = 0;

    virtual ~CASNode() = default;
};

class BasicCASNode : public CASNode {
public:
    virtual int compare(const BasicCASNode* other) const = 0;
};

class ConstantNode : public BasicCASNode {
    double value;
public:
    ConstantNode(double value) : value(value) {}
    
    std::ostream& print(std::ostream& os) const override {
        return os << value;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        return os << value;
    }

    expr_value getExprValue() const override {
        return { true, value };
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherConstantNode = dynamic_cast<const ConstantNode*>(other);
        assert(otherConstantNode != nullptr);
        if (value < otherConstantNode->value)
            return -1;
        if (value == otherConstantNode->value)
            return 0;
        return +1;
    }

    bool equals(const CASNode* other) const override {
        if (auto otherConstantNode = dynamic_cast<const ConstantNode*>(other))
            return (otherConstantNode->value == value);
        return false;
    }

    int getSortPriority() const override { return 0; }

    Polynomial toPolynomial() const override;
};

class VariableNode : public BasicCASNode {
    std::string name;
public:
    VariableNode(const std::string& name) : name(name) {}

    std::ostream& print(std::ostream& os) const override {
        return os << name;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        return os << name;
    }

    expr_value getExprValue() const override {
        return { false };
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherVariableNode = dynamic_cast<const VariableNode*>(other);
        assert(otherVariableNode != nullptr);

        return name.compare(otherVariableNode->name);
    }

    bool equals(const CASNode* other) const override {
        if (auto otherVariableNode = dynamic_cast<const VariableNode*>(other))
            return (otherVariableNode->name == name);
        return false;
    }

    int getSortPriority() const override { return 10; }

    Polynomial toPolynomial() const override;
};

class CosineNode : public BasicCASNode {
    std::shared_ptr<BasicCASNode> node;
public:
    CosineNode(std::shared_ptr<BasicCASNode> node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "cos(";
        node->print(os);
        os << ")";
        return os;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        os << "\\cos(";
        node->print(os);
        os << ")";
        return os;
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::cos(nodeValue.value) };
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        assert(otherCosineNode != nullptr);
        return node->compare(otherCosineNode->node.get());
    }

    bool equals(const CASNode* other) const override {
        auto otherCosineNode = dynamic_cast<const CosineNode*>(other);
        if (otherCosineNode == nullptr)
            return false;
        return (node->equals(otherCosineNode->node.get()));
    }

    int getSortPriority() const override { return 20; }

    Polynomial toPolynomial() const override;

};

class SineNode : public BasicCASNode {
    std::shared_ptr<BasicCASNode> node;
public:
    SineNode(std::shared_ptr<BasicCASNode> node) : node(node) {}

    std::ostream& print(std::ostream& os) const override {
        os << "sin(";
        node->print(os);
        os << ")";
        return os;
    }

    std::ostream& printLaTeX(std::ostream& os) const override {
        os << "\\sin(";
        node->print(os);
        os << ")";
        return os;
    }

    int compare(const BasicCASNode* other) const override {
        if (getSortPriority() < other->getSortPriority())
            return -1;
        if (getSortPriority() > other->getSortPriority())
            return +1;
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        assert(otherSineNode != nullptr);
        return node->compare(otherSineNode->node.get());
    }

    bool equals(const CASNode* other) const override {
        auto otherSineNode = dynamic_cast<const SineNode*>(other);
        if (otherSineNode == nullptr)
            return false;
        return (node->equals(otherSineNode->node.get()));
    }

    expr_value getExprValue() const override {
        auto nodeValue = node->getExprValue();
        if (!nodeValue.isConstant)
            return { false };
        return { true, std::sin(nodeValue.value) };
    }

    int getSortPriority() const override { return 30; }

    Polynomial toPolynomial() const override;
};

class Polynomial : public CASNode {
public:
    struct monomial_t {
        struct power_t {
            std::shared_ptr<BasicCASNode> base;
            int exponent = 1;
        };
        double coef = 1.0;
        std::vector<power_t> powers = {};

        int order() const {
            int sum = 0;
            for (const auto& p : powers)
                sum += p.exponent;
            return sum;
        }
    };

private:
    /// @brief monomial comparison function, strict order. return a < b 
    /// @return a < b. Happens when (1). a has less terms than b does, or (2). 
    /// the order of a is less than the order of b, or (3) 
    static bool monomial_cmp(const monomial_t& a, const monomial_t& b) {
        auto aSize = a.powers.size();
        auto bSize = b.powers.size();
        if (aSize < bSize) return true;
        if (aSize > bSize) return false;
        auto aOrder = a.order();
        auto bOrder = b.order();
        if (aOrder < bOrder) return true;
        if (aOrder > bOrder) return false;
        for (unsigned i = 0; i < aSize; i++) {
            int r = a.powers[i].base->compare(b.powers[i].base.get());
            if (r < 0) return true;
            if (r > 0) return false;
            if (a.powers[i].exponent > b.powers[i].exponent)
                return true;
            if (a.powers[i].exponent < b.powers[i].exponent)
                return false;
        }
        return false;
    };

    static bool monomial_eq(const monomial_t& a, const monomial_t& b) {
        auto aSize = a.powers.size();
        auto bSize = b.powers.size();
        if (aSize != bSize)
            return false;
        if (a.order() != b.order())
            return false;
        for (unsigned i = 0; i < aSize; i++) {
            if (a.powers[i].exponent != b.powers[i].exponent)
                return false;
            if (!(a.powers[i].base->equals(b.powers[i].base.get())))
                return false;
        }
        return true;
    }

    std::vector<monomial_t> monomials;

    Polynomial& operator+=(const monomial_t& monomial);
    
    Polynomial& operator-=(const monomial_t& monomial);

    Polynomial& operator*=(const monomial_t& monomial);
    
    void insertMonomial(const monomial_t& monomial) {
        auto it = std::lower_bound(monomials.begin(), monomials.end(), monomial, monomial_cmp);
        monomials.insert(it, monomial);
    }
public:
    Polynomial() : monomials() {}
    Polynomial(double v) : monomials({{v, {}}}) {}
    Polynomial(std::initializer_list<monomial_t> monomials)
        : monomials(monomials) {}

    std::ostream& print(std::ostream& os) const override;

    std::ostream& printLaTeX(std::ostream& os) const override;

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly) {
        return poly.print(os);
    }

    bool equals(const CASNode* other) const override {
        return false;
    }

    expr_value getExprValue() const override {
        double v = 0.0;
        double mV = 1.0;
        for (const auto& m : monomials) {
            mV = 1.0;
            for (const auto& p : m.powers) {
                auto baseV = p.base->getExprValue();
                if (!baseV.isConstant)
                    return { false };
                mV *= std::pow(baseV.value, p.exponent);
            }
            v += mV;
        }
        return { true, v };
    }

    int getSortPriority() const override { return 60; }

    Polynomial toPolynomial() const override { return Polynomial(*this); };

    Polynomial& operator+=(const Polynomial& other);

    Polynomial operator+(const Polynomial& other) const {
        // TODO: a better method
        Polynomial newPoly(*this);
        return newPoly += other;
    }

    Polynomial& operator-=(const Polynomial& other);

    Polynomial operator-(const Polynomial& other) const {
        // TODO: a better method
        Polynomial newPoly(*this);
        return newPoly -= other;
    }

    Polynomial& operator*=(const Polynomial& other);

    Polynomial operator*(const Polynomial& other) const;
};

template<typename real_ty>
class Complex {
public:
    real_ty real, imag;
    Complex() : real(), imag() {}
    Complex(real_ty real, real_ty imag) : real(real), imag(imag) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imzg);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    Complex& operator*=(const Complex& other) {
        real_ty r = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        real = r;
        return *this;
    }
};

class GateMatrix {
public:
    unsigned nqubits;
    size_t N;
    std::vector<Complex<Polynomial>> matrix;

    GateMatrix() : nqubits(0), N(1), matrix(1) {}
    GateMatrix(unsigned nqubits)
        : nqubits(nqubits), N(1 << nqubits), matrix(1 << (nqubits*2)) {}
    GateMatrix(std::initializer_list<Complex<Polynomial>> matrix)
        : matrix(matrix) { int r = updateNQubits(); assert(r > 0); }

    /// @brief update nqubits and N based on matrix.
    /// @return if matrix represents a (2**n) * (2**n) matrix, then return n
    /// (number of qubits). Otherwise, return -1 if matrix is empty; -2 if
    /// matrix.size() is not a perfect square; -3 if matrix represents an N * N
    /// matrix, but N is not a power of two.
    int updateNQubits() {
        if (matrix.empty())
            return -1;
        size_t size = static_cast<size_t>(std::sqrt(matrix.size()));
        if (size * size == matrix.size()) {
            N = size;
            if ((N & (N-1)) == 0) {
                nqubits = static_cast<unsigned>(std::log2(N));
                return nqubits;
            }
            return -3;
        }
        return -2;
    }
    
    bool checkSizeMatch() const {
        if (N != (1 << nqubits))
            return false;
        if (matrix.size() != N * N)
            return false;
        return true;
    }

    static GateMatrix Identity(unsigned nqubits) {
        GateMatrix m {nqubits};
        for (size_t r = 0; r < m.N; r++)
            m.matrix[r*m.N + r].real = { 1.0 };
        return m;
    }

    GateMatrix matmul(const GateMatrix& other) const {
        assert(checkSizeMatch());
        assert(other.checkSizeMatch());
        assert(nqubits == other.nqubits && N == other.N);

        GateMatrix m(nqubits);
        for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
            // C_{ij} = A_{ik} B_{kj}
            m.matrix[i*N + j] += matrix[i*N + k] * other.matrix[k*N + j];
        } } }
        return m;
    }

    GateMatrix kron(const GateMatrix& other) const {
        assert(checkSizeMatch());
        assert(other.checkSizeMatch());

        size_t lsize = N;
        size_t rsize = other.N;
        size_t msize = lsize * rsize;
        GateMatrix m(nqubits + other.nqubits);
        for (size_t lr = 0; lr < lsize; lr++) {
        for (size_t lc = 0; lc < lsize; lc++) {
        for (size_t rr = 0; rr < rsize; rr++) {
        for (size_t rc = 0; rc < rsize; rc++) {
            size_t r = lr * rsize + rr;
            size_t c = lc * rsize + rc;
            m.matrix[r*msize + c] = matrix[lr*lsize + lc] * other.matrix[rr*rsize + rc];
        } } } }
        return m;
    }

    GateMatrix leftKronI() const {
        assert(checkSizeMatch());

        GateMatrix m(2 * nqubits);
        for (size_t i = 0; i < N; i++) {
        for (size_t r = 0; r < N; r++) {
        for (size_t c = 0; c < N; c++) {
            m.matrix[(i*N + r) * N * N + (i*N + c)] = matrix[r*N + c];
        } } }
        return m;
    }

    GateMatrix rightKronI() const {
        assert(checkSizeMatch());

        GateMatrix m(2 * nqubits);
        for (size_t i = 0; i < N; i++) {
        for (size_t r = 0; r < N; r++) {
        for (size_t c = 0; c < N; c++) {
            m.matrix[(r*N + i) * N * N + (c*N + i)] = matrix[r*N + c];
        } } }
        return m;
    }

    GateMatrix swapTargetQubits() const {
        assert(nqubits == 2 && N == 4);

        return {{matrix[ 0], matrix[ 2], matrix[ 1], matrix[ 3],
                 matrix[ 8], matrix[10], matrix[ 9], matrix[11],
                 matrix[ 4], matrix[ 6], matrix[ 5], matrix[ 7],
                 matrix[12], matrix[14], matrix[13], matrix[15]}};
    }

    std::ostream& print(std::ostream& os) const {
        for (size_t r = 0; r < N; r++) {
            for (size_t c = 0; c < N; c++) {
                const auto& re = matrix[r*N + c].real;
                const auto& im = matrix[r*N + c].imag;
                re.print(os) << " + ";
                im.print(os) << "i ";
            }
            os << "\n";
        }
        return os;
    }

};

} // namespace quench::cas

#endif // QUENCH_GATE_MATRIX_H