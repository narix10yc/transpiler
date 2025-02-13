#ifndef UTILS_TASKDISPATCHER_H
#define UTILS_TASKDISPATCHER_H

#include <queue>
#include <thread>

namespace utils {

/// Thread-safe task dispatcher
/// TODO: We should separate syncing and joining the threads.
/// Could introduce another status called "Synced"
class TaskDispatcher {
  std::queue<std::function<void()>> tasks;
  std::vector<std::thread> workers;
  int nActiveTasks;
  std::mutex mtx;
  std::condition_variable cv;
  std::condition_variable syncCV;

  bool isStopped;

  // Fetch a task from the queue
  bool dequeue(std::function<void()>& task);
public:
  TaskDispatcher(int nWorkers);

  TaskDispatcher(const TaskDispatcher&) = delete;
  TaskDispatcher(TaskDispatcher&&) = delete;

  TaskDispatcher& operator=(const TaskDispatcher&) = delete;
  TaskDispatcher& operator=(TaskDispatcher&&) = delete;

  ~TaskDispatcher() {
    if (!isStopped)
      sync();
  }

  // Add a new task to the queue
  void enqueue(std::function<void()> task);

  int getWorkerID(std::thread::id threadID) const;

  /// @brief Sync and join all workers
  void sync();

}; // TaskDispatcher

} // namespace utils

#endif // UTILS_TASKDISPATCHER_H
