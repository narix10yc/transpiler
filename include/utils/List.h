#ifndef UTILS_LIST_H
#define UTILS_LIST_H

#include "utils/IteratorBase.h"
#include <cassert>
#include <memory>

#include <list>
namespace utils {

/// A double-link list template
template<typename T>
class List {
public:
  struct Node {
    T data;
    Node* prev;
    Node* next;
  }; // Node
private:
  class Iterator : public impl::IteratorBase<Iterator, T> {
    Node* current;
  public:
    explicit Iterator(Node* node) : current(node) {}

    Node* raw_ptr() const { return current; }

    T& operator*() const {
      assert(current != nullptr);
      return current->data;
    }

    T* operator->() const {
      assert(current != nullptr);
      return &(current->data);
    }

    Iterator& increment() {
      assert(current != nullptr);
      current = current->next;
      return *this;
    }

    Iterator& decrement() {
      assert(current != nullptr);
      current = current->prev;
      return *this;
    }

    Iterator next() const { assert(current); return Iterator(current->next); }
    Iterator prev() const { assert(current); return Iterator(current->prev); }

    bool equals(const Iterator& other) const {
      return current == other.current;
    }

    bool operator==(std::nullptr_t) const { return current == nullptr; }
    explicit operator bool() const { return current != nullptr; }
    explicit operator Node*() const { return current; }
  }; // Iterator

  class ConstIterator : public impl::IteratorBase<ConstIterator, T> {
    Node* current;
  public:
    explicit ConstIterator(Node* node) : current(node) {}
    ConstIterator(const Iterator& iter) : current(iter.raw_ptr()) {}

    Node* raw_ptr() const { return current; }

    const T& operator*() const {
      assert(current != nullptr);
      return current->data;
    }

    const T* operator->() const {
      assert(current != nullptr);
      return &(current->data);
    }

    ConstIterator& increment() {
      assert(current != nullptr);
      current = current->next;
      return *this;
    }

    ConstIterator& decrement() {
      assert(current != nullptr);
      current = current->prev;
      return *this;
    }

    ConstIterator next() const {
      assert(current);
      return ConstIterator(current->next);
    }

    ConstIterator prev() const {
      assert(current);
      return ConstIterator(current->prev);
    }

    bool equals(const ConstIterator& other) const {
      return current == other.current;
    }

    bool operator==(std::nullptr_t) const { return current == nullptr; }
    explicit operator bool() const { return current != nullptr; }
    explicit operator Node*() const { return current; }
  }; // ConstIterator

  Node* _head;
  Node* _tail;
  size_t _size;
public:
  using iterator = Iterator;
  using const_iterator = ConstIterator;

  List() : _head(nullptr), _tail(nullptr), _size(0) {}

  ~List() {
    while (_head != nullptr)
      pop_front();
  }

  List(const List& other) {
    for (const auto& data : other)
      push_back(data);
  }

  List(List&& other) noexcept
    : _head(other._head), _tail(other._tail), _size(other._size) {
    other._head = nullptr;
    other._tail = nullptr;
  }

  List& operator=(const List& other) {
    if (this == &other)
      return *this;
    this->~List();
    new (this) List(other);
    return *this;
  }

  List& operator=(List&& other) noexcept{
    if (this == &other)
      return *this;
    this->~List();
    new (this) List(std::move(other));
    return *this;
  }

  Node* head() { return _head; }
  const Node* head() const { return _head; }

  iterator head_iter() { return iterator(_head); }
  const_iterator head_iter() const { return const_iterator(_head); }
  const_iterator head_citer() const { return const_iterator(_head); }

  Node* tail() { return _tail; }
  const Node* tail() const { return _tail; }

  iterator tail_iter() { return iterator(_tail); }
  const_iterator tail_iter() const { return const_iterator(_tail); }
  const_iterator tail_citer() const { return const_iterator(_tail); }

  bool empty() const { return _size == 0; }

  size_t size() const { return _size; }

  void push_front(const T& value) { insert(_head, value); }

  void push_back(const T& value) { insert(nullptr, value); }

  /// construct and insert an element to the front of the list
  template<typename... Args>
  void emplace_front(Args&&... args) {
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(std::forward<Args>(args)...); // placement new
    node->prev = nullptr;
    node->next = _head;
    if (_tail == nullptr)
      _tail = node;
    else
      _head->prev = node;
    _head = node;
  }

  /// construct and insert an element to the back of the list
  template<typename... Args>
  void emplace_back(Args&&... args) {
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(std::forward<Args>(args)...); // placement new
    node->prev = _tail;
    node->next = nullptr;
    if (_head == nullptr) {
      assert(_tail == nullptr);
      assert(_size == 1);
      _head = node;
    }
    else
      _tail->next = node;
    _tail = node;
  }

  /// destruct the first element and pop it from the list
  void pop_front() {
    assert(!empty());
    if (empty()) // safeguard
      return;
    --_size;
    auto* tmp = _head;
    _head = tmp->next;
    tmp->data.~T();
    ::operator delete(tmp);
    if (_head == nullptr) {
      assert(empty());
      _tail = nullptr;
    }
    else
      _head->prev = nullptr;
  }

  /// destruct the last element and pop it from the list
  void pop_back() {
    assert(!empty());
    if (empty()) // safeguard
      return;
    --_size;
    auto* tmp = _tail;
    _tail = tmp->prev;
    tmp->data.~T();
    ::operator delete(tmp);
    if (_tail == nullptr) {
      assert(empty());
      _head = nullptr;
    }
    else
      _tail->next = nullptr;
  }

  iterator begin() { return iterator(_head); }
  iterator end() { return iterator(nullptr); }

  const_iterator begin() const { return const_iterator(_head); }
  const_iterator end() const { return const_iterator(nullptr); }

  const_iterator cbegin() const { return const_iterator(_head); }
  const_iterator cend() const { return const_iterator(nullptr); }

  Node* insert(Node* at, const T& value) {
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(value); // copy constructor
    if (at == nullptr) {
      node->prev = _tail;
      if (_tail != nullptr)
        _tail->next = node;
      node->next = nullptr;
      _tail = node;
      return node;
    }
    node->prev = at->prev;
    node->next = at;
    if (at->prev != nullptr)
      at->prev->next = node;
    at->prev = node;
    return node;
  }

  iterator insert(const_iterator it, const T& value) {
    return iterator(insert(static_cast<Node*>(it), value));
  }

  template<typename... Args>
  Node* emplace_insert(Node* at, Args&&... args) {
    ++_size;
    Node* node = static_cast<Node*>(::operator new(sizeof(Node)));
    new (&node->data) T(std::forward<Args>(args)...); // placement new
    if (at == nullptr) {
      node->prev = _tail;
      if (_tail != nullptr)
        _tail->next = node;
      node->next = nullptr;
      _tail = node;
      return node;
    }
    node->prev = at->prev;
    node->next = at;
    if (at->prev != nullptr)
      at->prev->next = node;
    at->prev = node;
    return node;
  }

  template<typename... Args>
  iterator emplace_insert(const_iterator it, Args&&... args) {
    return iterator(emplace_insert(
      const_cast<Node*>(static_cast<Node*>(it)),
      std::forward<Args>(args)...));
  }

  Node* erase(Node* at) {
    assert(at != nullptr);
    --_size;
    if (at->prev != nullptr)
      at->prev->next = at->next;
    if (at->next != nullptr)
      at->next->prev = at->prev;
    if (at == _head)
      _head = at->next;
    if (at == _tail)
      _tail = at->prev;
    auto* tmp = at->next;
    at->data.~T();
    ::operator delete(at);
    return tmp;
  }

  iterator erase(const_iterator it) {
    return iterator(erase(static_cast<Node*>(it)));
  }


};

} // namespace utils

#endif // UTILS_LIST_H
