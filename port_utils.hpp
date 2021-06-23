#ifndef __PORT_UTILS__
#define __PORT_UTILS__

/*
#ifndef IS_11
  #define IS_11
#endif
*/
#ifdef IS_11
  namespace std {
    template<typename T, typename... Args>
    unique_ptr<T> make_unique(Args&&... params) {
      return unique_ptr<T>(new T(std::forward<Args>(params)...));
    }
  }
#endif

#endif
