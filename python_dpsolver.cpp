#include "python_dpsolver.hpp"

using namespace Objectives;

#undef MULT_CLUST
#undef RISK_PART

#define GAUSS_OBJ objective_fn::Gaussian
#define POISS_OBJ objective_fn::Poisson
#define RATL_OBJ  objective_fn::RationalScore

#define DPSOLVER_RISK_PART_(n,T,a,b)  (DPSolver(n, T, a, b, objective_fn::Gaussian, true, false))
#define DPSOLVER_MULT_CLUST_(n,T,a,b) (DPSolver(n, T, a, b, objective_fn::Gaussian, false, true))

#ifdef MULT_CLUST
#define DPSOLVER_(n,T,a,b) (DPSOLVER_MULT_CLUST_(n,T,a,b))
#elseif RISK_PART
#define DPSOLVER_(n,T,a,b) (DPSOLVER_RISK_PART_(n,T,a,b))
#endif

std::vector<std::vector<int> > find_optimal_partition__DP(int n,
							 int T,
							 std::vector<float> a,
							 std::vector<float> b,
							 int parametric_dist,
							 bool risk_partitioning_objective,
							 bool use_rational_optimization) {
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization);
  return dp.get_optimal_subsets_extern();
}

float find_optimal_score__DP(int n,
			     int T,
			     std::vector<float> a,
			     std::vector<float> b,
			     int parametric_dist,
			     bool risk_partitioning_objective,
			     bool use_rational_optimization) {
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization);
  return dp.get_optimal_score_extern();
}

std::pair<std::vector<std::vector<int> >, float> optimize_one__DP(int n,
			int T,
			std::vector<float> a,
			std::vector<float> b,
			int parametric_dist,
			bool risk_partitioning_objective,
			bool use_rational_optimization) {
  auto dp = DPSolver(n, 
		     T, 
		     a, 
		     b, 
		     static_cast<objective_fn>(parametric_dist), 
		     risk_partitioning_objective, 
		     use_rational_optimization);
  std::vector<std::vector<int> > subsets = dp.get_optimal_subsets_extern();
  float score = dp.get_optimal_score_extern();
  
  return std::make_pair(subsets, score);
}
