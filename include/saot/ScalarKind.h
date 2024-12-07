#ifndef SAOT_SCALARKIND_H
#define SAOT_SCALARKIND_H

namespace saot {

enum ScalarKind : int {
  SK_Zero = 0,
  SK_One = 1,
  SK_MinusOne = -1,
  SK_General = 2,
  SK_ImmValue = 3,
  SK_Shared = 4,
  SK_SharedNeg = 5,
};

}

#endif // SAOT_SCALARKIND_H