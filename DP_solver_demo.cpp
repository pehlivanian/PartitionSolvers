#include "DP_solver_demo.hpp"

void print_subsets(std::vector<std::vector<int> >& subsets) {
  std::cout << "SUBSETS\n";
  std::cout << "[\n";
  std::for_each(subsets.begin(), subsets.end(), [](std::vector<int>& subset){
		  std::cout << "[";
		  std::copy(subset.begin(), subset.end(),
			    std::ostream_iterator<int>(std::cout, " "));
		  std::cout << "]\n";
		});
  std::cout << "]";
}

auto main() -> int {
 
  constexpr int n = 10;
  constexpr int T = 3;
  constexpr int NUM_TRIALS = 10;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<float> dista(-10., 10.);
  std::uniform_real_distribution<float> distb(0., 10.);

  auto gena = [&dista, &mersenne_engine]() { return dista(mersenne_engine); };
  auto genb = [&distb, &mersenne_engine]() { return distb(mersenne_engine); };

  std::vector<float> a(n), b(n);

  for (int i=0; i<NUM_TRIALS; ++i) {
    
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);
    
    auto dp = DPSolver(n, T, a, b);
    auto dp_opt = dp.get_optimal_subsets_extern();
    auto dp_scores = dp.get_score_by_subset_extern();
    auto dp_score = dp.get_optimal_score_extern();

    auto dp_pen = DPSolver(n, T, a, b, 
			   objective_fn::Gaussian, 
			   false,
			   false,
			   1.0,
			   2);
    auto dp_pen_opt = dp_pen.get_optimal_subsets_extern();
    auto dp_pen_scores = dp_pen.get_score_by_subset_extern();
    auto dp_pen_score = dp_pen.get_optimal_score_extern();
    
    auto ltss = LTSSSolver(n, a, b);
    auto ltss_opt = ltss.get_optimal_subset_extern();
    auto ltss_score = ltss.get_optimal_score_extern();
    
    std::cout << "\n========\nTRIAL: " << i << "\n========\n";
    std::cout << "\na: { ";
    std::copy(a.begin(), a.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "}" << std::endl;
    std::cout << "b: { ";
    std::copy(b.begin(), b.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "}" << std::endl;
    std::cout << "\nDPSolver subsets:\n";
    std::cout << "================\n";
    print_subsets(dp_opt);
    std::cout << "\nSubset scores: ";
    std::copy(dp_scores.begin(), dp_scores.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "\nPartition score: ";
    std::cout << dp_score << std::endl;
    std::cout << "\n\nDPSolver with penaly subsets:\n";
    std::cout << "================\n";
    print_subsets(dp_pen_opt);
    std::cout << "\nSubset scores: ";
    std::copy(dp_pen_scores.begin(), dp_pen_scores.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "\nPartition score: ";
    std::cout << dp_pen_score << std::endl;
    std::cout << "\nLTSSSolver subset:\n";
    std::cout << "=================\n";
    std::copy(ltss_opt.begin(), ltss_opt.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";
    std::cout << "\nScore: ";
    std::cout << ltss_score << "\n";
  }

  return 0;
}
