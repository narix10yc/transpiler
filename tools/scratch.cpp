#include "utils/square_matrix.h"
#include "utils/utils.h"

int main() {
  auto randomMat = utils::randomUnitaryMatrix(4);

  utils::printComplexMatrixF64(randomMat);
  return 0;
}