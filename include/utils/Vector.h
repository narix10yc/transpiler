#ifndef UTILS_VECTOR_H
#define UTILS_VECTOR_H

#include <cassert>
#include <memory>
#include <iostream>

namespace utils {

template<typename T>
class Vector {
  T* _data;
  size_t _capacity;
  size_t _size;

public:
  /// This constructor will NOT initialize array
  explicit Vector(size_t capacity = 1)
    : _data(static_cast<T*>(::operator new(sizeof(T) * capacity)))
    , _capacity(capacity)
    , _size(0) {
    // std::cerr << "Vector<" << typeid(T).name()
              // << "> initialized with capacity " << capacity
              // << " @ " << _data << "\n";
  }

  Vector(size_t size, const T& value)
    : _data(static_cast<T*>(::operator new(sizeof(T) * size)))
    , _capacity(size)
    , _size(size) {
    for (size_t i = 0; i < size; i++)
      new (_data + i) T(value); // copy constructor
  }

  ~Vector() {
    for (size_t i = 0; i < _size; i++)
      _data[i].~T();
    ::operator delete(_data);
  }

  Vector(const Vector& other)
    : _data(static_cast<T*>(::operator new(other.capacityInBytes())))
    , _capacity(other._capacity)
    , _size(other._size) {
    for (size_t i = 0; i < _size; ++i)
      new (_data + i) T(other[i]); // copy constructor
  }

  Vector(Vector&& other) noexcept {
    _capacity = other._capacity;
    _size = other._size;
    _data = other._data;
    other._data = nullptr;
  }

  Vector& operator=(const Vector& other) {
    if (this == &other)
      return *this;
    _capacity = other._capacity;
    _size = other._size;
    _data = static_cast<T*>(::operator new(sizeof(T) * _capacity));
    for (size_t i = 0; i < _size; ++i)
      _data[i] = T(other[i]); // copy assignment
    return *this;
  }

  Vector& operator=(Vector&& other) noexcept {
    if (this == &other)
      return *this;
    _capacity = other._capacity;
    _size = other._size;
    _data = other._data;
    other._data = nullptr;
    return *this;
  }

  size_t size() const { return _size; }
  size_t capacity() const { return _capacity; }

  size_t sizeInBytes() const { return sizeof(T) * _size; }
  size_t capacityInBytes() const { return sizeof(T) * _capacity; }

  bool empty() const { return _size == 0; }

  T& operator[](size_t index) {
    assert(index < _size);
    return _data[index];
  }

  const T& operator[](size_t index) const {
    assert(index < _size);
    return _data[index];
  }

  T* begin() { return _data; }
  const T* begin() const { return _data; }
  const T* cbegin() const { return _data; }

  T* end() { return _data + _size; }
  const T* end() const { return _data + _size; }
  const T* cend() const { return _data + _size; }

  T& front() { assert(_size > 0); return _data[0]; }
  const T& front() const { assert(_size > 0); return _data[0]; }

  T& back() { assert(_size > 0); return _data[_size - 1]; }
  const T& back() const { assert(_size > 0); return _data[_size - 1]; }

  void push_back(const T& item) {
    if (_size == _capacity)
      reserve(_capacity << 1);
    _data[_size++] = item;
  }

  void push_back(T&& item) {
    if (_size == _capacity)
      reserve(_capacity << 1);
    _data[_size++] = std::move(item);
  }

  template<typename... Args>
  void emplace_back(Args&&... args) {
    if (_size == _capacity)
      reserve(_capacity << 1);
    new (_data + _size++) T(std::forward<Args>(args)...);
  }

  // Destroy the last element and reduce size by 1.
  void pop_back() {
    assert(_size > 0);
    back().~T();
    --_size;
  }

  // Pop the last element and move assign it into \p obj.
  void pop_back_and_collect(T& obj) {
    assert(_size > 0);
    obj = std::move(_data[_size - 1]);
    pop_back();
  }

  void reserve(size_t capacity) {
    if (capacity <= _capacity)
      return;
    T* newData = static_cast<T*>(::operator new(sizeof(T) * capacity));
    for (size_t i = 0; i < _size; ++i)
      new (newData + i) T(std::move(_data[i]));
    delete[] _data;
    _data = newData;
    _capacity = capacity;
  }

};

}

#endif // UTILS_VECTOR_H
