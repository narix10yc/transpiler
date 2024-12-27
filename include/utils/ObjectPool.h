#ifndef UTILS_OBJECTPOOL_H
#define UTILS_OBJECTPOOL_H

#include "utils/PODVector.h"
#include <iostream>

namespace utils {

/// Simple implementation of object pooling.
template<typename T, size_t block_size = 128>
class ObjectPool {
  PODVector<T*> objHolders;
  PODVector<T*> availables;

  void extendPool() {
    T* data = static_cast<T*>(::operator new(sizeof(T) * block_size));
    objHolders.push_back(data);
    for (size_t i = 0; i < block_size; ++i)
      availables.push_back(data + i);
  }

public:
  ObjectPool() : objHolders(), availables() { extendPool(); }

  ~ObjectPool() {
    for (T* obj : objHolders) {
      for (size_t i = 0; i < block_size; ++i) {
        bool inited = true;
        for (const T* item : availables) {
          if (item == obj + i) {
            inited = false;
            break;
          }
        }
        if (inited)
          (obj + i)->~T();
      }
      ::operator delete(obj);
    }
  }

  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;
  ObjectPool(ObjectPool&&) = delete;
  ObjectPool& operator=(ObjectPool&&) = delete;

  /// acquire an instance from the pool and construct it in-place
  template<typename... Args>
  T* acquire(Args&&... args) {
    if (availables.empty())
      extendPool();

    assert(!availables.empty());
    T* obj = availables.back();
    availables.pop_back();

    new (obj) T(std::forward<Args>(args)...);
    return obj;
  }

  /// destroy an object and put back into the pool
  void release(T* obj) {
    assert(isInPool(obj) &&
      "Trying to release an object that is not managed by this pool");
    obj->~T();
    availables.push_back(obj);
  }

  /// Check if obj is in managed by this pool
  bool isInPool(const T* obj) {
    for (const T* ptr : objHolders) {
      if (ptr <= obj && obj < ptr + block_size)
        return true;
    }
    std::cerr << "ObjectPool<" << typeid(T).name() << "> @ " << this
              << " is currently managing " << objHolders.size() << " segments:\n";
    for (const T* ptr : objHolders)
      std::cerr << "  [" << ptr << ", " << ptr + block_size << ")\n";
    std::cerr << "But obj @ " << obj << " is not in Pool.\n";
    return false;
  }

};

} // namespace utils


#endif // UTILS_OBJECTPOOL_H
