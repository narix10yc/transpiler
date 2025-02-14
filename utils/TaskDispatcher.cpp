#include "utils/TaskDispatcher.h"

#include <iostream>
#include <utils/utils.h>

using namespace utils;

void TaskDispatcher::enqueue(const std::function<void()>& task) {
  {
    std::lock_guard lock(mtx);
    ++nTotalTasks;
    tasks.push(std::move(task));
  }
  cv.notify_one();
}

void TaskDispatcher::workerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock lock(mtx);
      cv.wait(lock, [this]() {
        return status != Running || !tasks.empty();
      });

      if (status != Running && tasks.empty())
        return;

      task = std::move(tasks.front());
      tasks.pop();
    }
    ++nActiveWorkers;
    task();
    --nActiveWorkers;
    syncCV.notify_all();
  }
}

TaskDispatcher::TaskDispatcher(int nWorkers)
  : tasks(), workers(), nTotalTasks(0), nActiveWorkers(0)
  , mtx(), cv(), syncCV(), status(Running) {
  workers.reserve(nWorkers);
  for (int i = 0; i < nWorkers; ++i) {
    workers.emplace_back([this]() {
      workerThread();
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

void TaskDispatcher::sync(bool progressBar) {
  {
    std::unique_lock lock(mtx);
    syncCV.wait(lock, [this, progressBar]() {
      // std::cerr << "nTasks/nActiveWorkers/nWorkers: "
                // << tasks.size() << "/" << nActiveWorkers << "/" << workers.size()
                // << "\n";
      if (progressBar)
        utils::displayProgressBar(nTotalTasks - tasks.size(), nTotalTasks, 50);
      return tasks.empty() && nActiveWorkers == 0;
    });
    if (progressBar)
      std::cout << std::endl;
    status = Synced;
  }
  cv.notify_all();
}

void TaskDispatcher::join() {
  for (auto& thread : workers) {
    if (thread.joinable())
      thread.join();
  }
  status = Stopped;
}