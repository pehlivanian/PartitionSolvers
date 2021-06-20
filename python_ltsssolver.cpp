#include "python_ltsssolver.hpp"

std::vector<int> find_optimal_partition__LTSS(int n,
				  std::vector<float> a,
				  std::vector<float> b) {

  auto ltss = LTSSSolver(n, a, b);
  return ltss.get_optimal_subset_extern();

}

float find_optimal_score__LTSS(int n,
			       std::vector<float> a,
			       std::vector<float> b) {
  auto ltss = LTSSSolver(n, a, b);
  return ltss.get_optimal_score_extern();
}

std::pair<std::vector<int>, float> optimize_one__LTSS(int n,
						      std::vector<float> a,
						      std::vector<float> b) {
  auto ltss = LTSSSolver(n, a, b);
  std::vector<int> subset = ltss.get_optimal_subset_extern();
  float score = ltss.get_optimal_score_extern();
  return std::make_pair(subset, score);
}
