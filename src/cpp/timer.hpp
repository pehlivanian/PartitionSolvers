#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <atomic>
#include <sstream>

template<typename Clock=std::chrono::high_resolution_clock>
class Timer {
public:
  Timer() : start_point(Clock::now()) {}
  template<typename Rep=typename Clock::duration::rep, typename Units=typename Clock::duration>
  Rep elapsed_time() const {
    std::atomic_thread_fence(std::memory_order_relaxed);
    auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
    std::atomic_thread_fence(std::memory_order_relaxed);
    return static_cast<Rep>(counted_time);
  }
private:
  const typename Clock::time_point start_point;
};

using precise_timer = Timer<>;
using system_timer = Timer<std::chrono::system_clock>;
using monotonic_timer = Timer<std::chrono::steady_clock>;

#endif
