#include "utils/TaskDispatcher.h"

#include <iostream>

using namespace utils;

void TaskDispatcher::enqueue(std::function<void()> task) {
  std::lock_guard<std::mutex> lock(mtx);
  tasks.push(std::move(task));
  cv.notify_one();
}

bool TaskDispatcher::dequeue(std::function<void()>& task) {
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [this]() {
    return isStopped || !tasks.empty();
  });

  if (isStopped) {
    assert(tasks.empty());
    return false;
  }

  task = std::move(tasks.front());
  tasks.pop();
  return true;
}

TaskDispatcher::TaskDispatcher(int nWorkers)
  : tasks(), workers(), nActiveTasks(0)
  , mtx(), cv(), syncCV(), isStopped(false) {
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i) {
    workers.emplace_back([this]() {
      while (true) {
        std::function<void()> task;
        if (!dequeue(task))
          break;
        {
          std::lock_guard<std::mutex> lock(mtx);
          ++nActiveTasks;
        }
        task();
        {
          std::lock_guard<std::mutex> lock(mtx);
          if (--nActiveTasks == 0)
            syncCV.notify_all();
        }
      }
    });
  }
}

int TaskDispatcher::getWorkerID(std::thread::id threadID) const {
  for (int n = workers.size(), i = 0; i < n; ++i) {
    if (workers[i].get_id() == threadID)
      return i;
  }
  assert(false && "Unreachable");
  return -1;
}

void TaskDispatcher::sync() {
  // std::cerr << "Syncing...\n";
  {
    std::unique_lock<std::mutex> lock(mtx);
    syncCV.wait(lock, [this]() {
      // std::cerr << "nTasks/nActiveTasks/nWorkers: "
                // << tasks.size() << "/" << nActiveTasks << "/" << workers.size()
                // << "\n";
      return tasks.empty() && nActiveTasks == 0;
    });
    isStopped = true;
    /// Notify all waiting threads that no more tasks will be added
    /// Safe to exit
    cv.notify_all();
  }
  // std::cerr << "Synced!\n";
  for (auto& t : workers) {
    if (t.joinable())
      t.join();
  }
}