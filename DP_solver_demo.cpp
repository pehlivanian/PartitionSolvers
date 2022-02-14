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

float mixture_of_uniforms(int n) {
  int bin = 1;
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<float> distmixer(0., 1.);
  std::uniform_real_distribution<float> dista(0., 1./static_cast<float>(n));
  
  float mixer = distmixer(mersenne_engine);

  while (bin < n) {
    if (mixer < static_cast<float>(bin)/static_cast<float>(n))
      break;
    ++bin;
  }
  return dista(mersenne_engine) + static_cast<float>(bin)-1.;
}


auto main() -> int {
 
  constexpr int n = 5000;
  constexpr int T = 20;
  constexpr int NUM_TRIALS = 0;
  constexpr int NUM_BEST_SWEEP_OLS_TRIALS = 100;
  constexpr int NUM_DISTRIBUTIONS_IN_MIX = 9;
    
  int cluster_sum = 0;

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<float> dista(-10., 10.);
  std::uniform_real_distribution<float> distb(1., 1.);

  // auto gena = [&dista, &mersenne_engine]() { return dista(mersenne_engine); };
  auto gena = []() { return mixture_of_uniforms(NUM_DISTRIBUTIONS_IN_MIX); };
  auto genb = [&distb, &mersenne_engine]() { return distb(mersenne_engine); };

  std::vector<float> a(n), b(n);

  for (int i=0; i<NUM_BEST_SWEEP_OLS_TRIALS; ++i) {
    std::generate(a.begin(), a.end(), gena);
    std::generate(b.begin(), b.end(), genb);

    auto dp = DPSolver(n, T, a, b,
		       objective_fn::Poisson,
		       true,
		       true,
		       0.0,
		       1.0,
		       false,
		       true);

    auto dp_opt = dp.get_optimal_subsets_extern();
    auto dp_scores = dp.get_score_by_subset_extern();
    int num_clusters = dp.get_optimal_num_clusters_OLS_extern();

    std::cout << "TRIAL: " << i << " optimal number of clusters: " << num_clusters << std::endl;
    cluster_sum += num_clusters;
    
  }
  std::cout << "CLUSTER COUNT: " 
	    << static_cast<float>(cluster_sum)/static_cast<float>(NUM_BEST_SWEEP_OLS_TRIALS) 
	    << std::endl;

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
