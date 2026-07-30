#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>

namespace mirage {
namespace search {

template <typename T>
class SearchStateManager {
public:
  SearchStateManager() = default;
  virtual ~SearchStateManager() = default;

  virtual void add_state(T const &c) = 0;
  virtual bool pop_state(T &c) = 0;
  virtual size_t size() const = 0;
};

template <typename T>
class GlobalWorkerQueueManager : public SearchStateManager<T> {
public:
  GlobalWorkerQueueManager(std::chrono::milliseconds timeout = 1000)
      : timeout(timeout){};

  void add_state(T const &c) override {
    std::lock_guard<std::mutex> lock(queue_mutex);
    search_queue.push(c);
    queue_cv.notify_one();
  }

  bool pop_state(T &c) override {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (!queue_cv.wait_for(
            lock, timeout, [&] { return !search_queue.empty(); })) {
      return false;
    }
    c = search_queue.front();
    search_queue.pop();
    return true;
  }

  size_t size() const override {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return search_queue.size();
  }

private:
  std::queue<T> search_queue;
  mutable std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::chrono::milliseconds timeout;
};

template <typename T>
class LocalConditionalManager : public SearchStateManager<T> {
public:
  LocalConditionalManager(std::function<bool(T const &)> const &stop_condition)
      : stop_condition(stop_condition) {}

  void add_state(T const &c) override {
    search_queue.push(c);
  }

  bool pop_state(T &c) override {
    if (search_queue.empty()) {
      return false;
    }
    if (stop_condition(search_queue.front())) {
      return false;
    }
    c = search_queue.front();
    search_queue.pop();
    return true;
  }

  bool pop_state_without_condition(T &c) {
    if (search_queue.empty()) {
      return false;
    }
    c = search_queue.front();
    search_queue.pop();
    return true;
  }

  size_t size() const override {
    return search_queue.size();
  }

private:
  std::queue<T> search_queue;
  std::function<bool(T const &)> stop_condition;
};

} // namespace search
} // namespace mirage
