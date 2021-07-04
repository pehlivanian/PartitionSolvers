#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iterator>

#include "score.hpp"
#include "LTSS.hpp"
#include "DP.hpp"

void sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);
  
  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
	    
}

auto main(int argc, char **argv) -> int {

  /*
    std::vector<float> a1{2.26851454, 2.86139335, 5.51314769, 6.84829739, 6.96469186, 7.1946897,
    9.80764198, 4.2310646};
    std::vector<float> b1{3.43178016, 3.92117518, 7.29049707, 7.37995406, 4.80931901, 4.38572245,
    3.98044255, 0.59677897};
  */

  std::vector<float> a1{7.86437457, 6.56143048, 9.28844611, 8.6244709, 6.98280455, 9.99120625,
      6.7904846, 8.1787752, 9.36647966, 7.8152298};
  std::vector<float> b1{8.4410853, 1.69356328, 2.80002414, 9.11591302, 2.45803365, 6.25261449,
      6.39727496, 5.96518835, 9.47400081, 3.63103151};

  sort_by_priority(a1, b1);

  auto ltss1 = LTSSSolver(10, a1, b1, objective_fn::Poisson);
  auto ltss1_opt = ltss1.get_optimal_subset_extern();
  
  auto dp1 = DPSolver(10, 4, a1, b1, objective_fn::Poisson, false, false);
  auto dp1_opt = dp1.get_optimal_subsets_extern();

  std::cout << "FINISHED" << std::endl;

  return 0.;
}
