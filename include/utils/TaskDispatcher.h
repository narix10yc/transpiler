#ifndef UTILS_TASKDISPATCHER_H
#define UTILS_TASKDISPATCHER_H

#include <cassert>
#include <queue>
#include <functional>
#include <thread>
#include <condition_variable>

namespace utils {

/// Thread-safe task dispatcher
class TaskDispatcher {
  enum Status {
    /// Not in use yet
    Idle,
    /// Running
    Running,
    /// All threads have finished running, but have not yet been joined
    Synced,
    /// All
    Stopped
  };
  std::queue<std::function<void()>> tasks;
  std::vector<std::thread> workers;
  std::mutex mtx;
  std::condition_variable cv;
  std::condition_variable syncCV;

  int nTotalTasks;
  std::atomic<int> nActiveWorkers;
  std::atomic<Status> status;

  void workerThread();
public:
  TaskDispatcher(int nWorkers);

  TaskDispatcher(const TaskDispatcher&) = delete;
  TaskDispatcher(TaskDispatcher&&) = delete;

  TaskDispatcher& operator=(const TaskDispatcher&) = delete;
  TaskDispatcher& operator=(TaskDispatcher&&) = delete;

  ~TaskDispatcher() {
    if (status == Running)
      sync();
    if (status == Synced)
      join();
    assert(status == Stopped);
  }

  // Add a new task to the queue
  void enqueue(const std::function<void()>& task);

  int getWorkerID(std::thread::id threadID) const;

  /// @brief A blocking method that waits until all tasks are finished.
  void sync(bool progressBar = false);

  /// @brief Join all threads. This method is automatically called upon
  /// destruction, so is normally not needed.
  void join();

}; // TaskDispatcher

} // namespace utils

#endif // UTILS_TASKDISPATCHER_H
