#include <random>

#include "LTSS_solver_demo.hpp"

void print_subset(std::vector<int>& subset) {
  std::cout << "SUBSET\n";
  std::cout << "[ ";
  std::copy(subset.begin(), subset.end(),
	    std::ostream_iterator<int>(std::cout, " "));
  std::cout << " ]\n\n";
}

auto main() -> int {

  constexpr int N = 50;
  constexpr int NUM_TRIALS = 10;

  float lower_limit_a = 0., upper_limit_a = 100.;
  float lower_limit_b = 0., upper_limit_b = 100.;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};
  std::uniform_real_distribution<float> dista{lower_limit_a, upper_limit_a};
  std::uniform_real_distribution<float> distb{lower_limit_b, upper_limit_b};

  auto gena = [&dista, &mersenne_engine]() { return dista(mersenne_engine); };
  auto genb = [&distb, &mersenne_engine]() { return distb(mersenne_engine); };

  std::vector<float> a(N), b(N);
  
  for (int i=0; i<NUM_TRIALS; ++i) {
    std::cerr << "TRIAL: " << i << std::endl;

    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    auto ltss = LTSSSolver(N, a, b);

    std::vector<int> subset = ltss.get_optimal_subset_extern();

    print_subset(subset);
  }

  return 0;
}
