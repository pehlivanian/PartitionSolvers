#include "python_dpsolver.hpp"

using namespace Objectives;

#undef MULT_CLUST
#undef RISK_PART

#define GAUSS_OBJ objective_fn::Gaussian
#define POISS_OBJ objective_fn::Poisson
#define RATL_OBJ  objective_fn::RationalScore

#define DPSOLVER_RISK_PART_(n,T,a,b)  (DPSolver(n, T, a, b, GAUSS_OBJ, true, false))
#define DPSOLVER_MULT_CLUST_(n,T,a,b) (DPSolver(n, T, a, b, GAUSS_OBJ, false, true))

#ifdef MULT_CLUST
#define DPSOLVER_(n,T,a,b) (DPSOLVER_MULT_CLUST_(n,T,a,b))
#elseif RISK_PART
#define DPSOLVER_(n,T,a,b) (DPSOLVER_RISK_PART_(n,T,a,b))
#endif

struct distributionException : public std::exception {
  const char* what() const throw () {
    return "Bad distributional assignment";
  };
};


int compute_optimal_num_clusters_OLS(int n,
				     int T,
				     std::vector<float> a,
				     std::vector<float> b,
				     int parametric_dist,
				     bool risk_partitioning_objective,
				     bool use_rational_optimization,
				     float gamma,
				     int reg_power) {
  
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization,
		     gamma,
		     reg_power,
		     false,
		     true);

  return dp.get_optimal_num_clusters_OLS_extern();

}
float compute_score(std::vector<float> a, 
		    std::vector<float> b, 
		    int parametric_dist,
		    bool risk_partitioning_objective,
		    bool use_rational_optimization) {

  int n = a.size();  
  auto obj_fn = static_cast<objective_fn>(parametric_dist);
  std::unique_ptr<ParametricContext> context;

  if (obj_fn == objective_fn::Gaussian) {
    context = std::make_unique<GaussianContext>(a, 
						b, 
						n, 
						risk_partitioning_objective,
						use_rational_optimization);
  }
  else if (obj_fn == objective_fn::Poisson) {
    context = std::make_unique<PoissonContext>(a, 
					       b, 
					       n,
					       risk_partitioning_objective,
					       use_rational_optimization);
  }
  else if (obj_fn == objective_fn::RationalScore) {
    context = std::make_unique<RationalScoreContext>(a, 
						     b, 
						     n,
						     risk_partitioning_objective,
						     use_rational_optimization);
  }
  else {
    throw distributionException();
  }

  return context->compute_score(0, n);  
}

std::vector<std::vector<int> > find_optimal_partition__DP(int n,
							  int T,
							  std::vector<float> a,
							  std::vector<float> b,
							  int parametric_dist,
							  bool risk_partitioning_objective,
							  bool use_rational_optimization,
							  float gamma,
							  int reg_power) {
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization,
		     gamma,
		     reg_power);
  return dp.get_optimal_subsets_extern();
}

float find_optimal_score__DP(int n,
			     int T,
			     std::vector<float> a,
			     std::vector<float> b,
			     int parametric_dist,
			     bool risk_partitioning_objective,
			     bool use_rational_optimization,
			     float gamma,
			     int reg_power) {
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization,
		     gamma,
		     reg_power);
  return dp.get_optimal_score_extern();
}

std::vector<std::pair<std::vector<std::vector<int> >, float> > optimize_all__DP(int n,
										int T,
										std::vector<float> a,
										std::vector<float> b,
										int parametric_dist,
										bool risk_partitioning_objective,
										bool use_rational_optimization,
										float gamma,
										int reg_power) {
  auto dp = DPSolver(n,
		     T,
		     a,
		     b,
		     static_cast<objective_fn>(parametric_dist),
		     risk_partitioning_objective,
		     use_rational_optimization,
		     gamma,
		     reg_power,
		     true);
  
  return dp.get_all_subsets_and_scores_extern();
}


std::pair<std::vector<std::vector<int> >, float> optimize_one__DP(int n,
								  int T,
								  std::vector<float> a,
								  std::vector<float> b,
								  int parametric_dist,
								  bool risk_partitioning_objective,
								  bool use_rational_optimization,
								  float gamma,
								  int reg_power) {
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization,
		     gamma,
		     reg_power);
  std::vector<std::vector<int> > subsets = dp.get_optimal_subsets_extern();
  float score = dp.get_optimal_score_extern();

  return std::make_pair(subsets, score);
}

std::pair<std::vector<std::vector<int> >, float> sweep_best_OLS__DP(int n,
								   int T,
								   std::vector<float> a,
								   std::vector<float> b,
								   int parametric_dist,
								   bool risk_partitioning_objective,
								   bool use_rational_optimization,
								   float gamma,
								   int reg_power) {
  auto dp = DPSolver(n,
		     T,
		     a,
		     b,
		     static_cast<objective_fn>(parametric_dist),
		     risk_partitioning_objective,
		     use_rational_optimization,
		     gamma,
		     reg_power,
		     false,
		     true);
  std::vector<std::vector<int> > subsets = dp.get_optimal_subsets_extern();
  float score = dp.get_optimal_score_extern();
  
  return std::make_pair(subsets, score);
  
}

std::pair<std::vector<std::vector<int> >,float> sweep_best__DP(int n,
							      int T,
							      std::vector<float> a,
							      std::vector<float> b,
							      int parametric_dist,
							      bool risk_partitioning_objective,
							      bool use_rational_optimization,
							      float gamma,
							      int reg_power) {
  
  float best_score = std::numeric_limits<float>::lowest(), score;
  std::vector<std::vector<int> > subsets;

  for (int i=T; i>1; --i) {
    auto dp = DPSolver(n,
		       i,
		       a,
		       b,
		       static_cast<objective_fn>(parametric_dist),
		       risk_partitioning_objective,
		       use_rational_optimization,
		       gamma,
		       reg_power);

    score = dp.get_optimal_score_extern();
    std::cout << "NUM_PARTITIONS: " << i << " SCORE: " << score << std::endl;
    if (score > best_score) {
      // std::cout << "FOUND BEST PARTITION: i = " << i << " score = " << score << std::endl;
      best_score = score;
      subsets = dp.get_optimal_subsets_extern();
    }
  }
  return std::make_pair(subsets, best_score);
}

std::vector<std::pair<std::vector<std::vector<int> >,float> > sweep_parallel__DP(int n,
									       int T,
									       std::vector<float> a,
									       std::vector<float> b,
									       int parametric_dist,
									       bool risk_partitioning_objective,
									       bool use_rational_optimization,
									       float gamma,
									       int reg_power) {

  ThreadsafeQueue<std::pair<std::vector<std::vector<int> >, float> > results_queue;

  auto task = [&results_queue](int n,
			       int i,
			       std::vector<float> a,
			       std::vector<float> b,
			       int parametric_dist,
			       bool risk_partitioning_objective,
			       bool use_rational_optimization,
			       float gamma,
			       int reg_power) {
    auto dp = DPSolver(n,
		       i,
		       a,
		       b,
		       static_cast<objective_fn>(parametric_dist),
		       risk_partitioning_objective,
		       use_rational_optimization,
		       gamma,
		       reg_power);
    results_queue.push(std::make_pair(dp.get_optimal_subsets_extern(),
				      dp.get_optimal_score_extern()));
  };
  
  std::vector<ThreadPool::TaskFuture<void> > v;

  for (int i=T; i>=1; --i) {
    v.push_back(DefaultThreadPool::submitJob(task, n, i, a, b, parametric_dist, risk_partitioning_objective, use_rational_optimization, gamma, reg_power));
  }	       
  for (auto& item : v) 
    item.get();
  
  std::pair<std::vector<std::vector<int> >, float> result;
  std::vector<std::pair<std::vector<std::vector<int> >, float> > results{static_cast<size_t>(T+1)};
  while (!results_queue.empty()) {
    bool valid = results_queue.waitPop(result);
    if (valid) {
      results[result.first.size()] = result;
    }
  }

  return results;

}


