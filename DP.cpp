#include <algorithm>
#include <iterator>
#include <numeric>
#include <limits>
#include <iostream>
#include <iomanip>
#include <utility>
#include <cmath>
#include <string>
#include <exception>

#include "DP.hpp"

struct distributionException : public std::exception {
  const char* what() const throw () {
    return "Bad distributional assignment";
  };
};

template<typename T>
class TD;

void
DPSolver::sort_by_priority(std::vector<float>& a, std::vector<float>& b) {
  std::vector<int> ind(a.size());
  std::iota(ind.begin(), ind.end(), 0);

  std::stable_sort(ind.begin(), ind.end(),
		   [&a, &b](int i, int j) {
		     return (a[i]/b[i]) < (a[j]/b[j]);
		   });

  priority_sortind_ = ind;

  // Inefficient reordering
  std::vector<float> a_s, b_s;
  for (auto i : ind) {
    a_s.push_back(a[i]);
    b_s.push_back(b[i]);
  }

  std::copy(a_s.begin(), a_s.end(), a.begin());
  std::copy(b_s.begin(), b_s.end(), b.begin());
}

void
DPSolver::print_maxScore_() {

  for (int i=0; i<n_; ++i) {
    std::copy(maxScore_[i].begin(), maxScore_[i].end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
  }
}

void
DPSolver::print_nextStart_() {
  for (int i=0; i<n_; ++i) {
    std::copy(nextStart_[i].begin(), nextStart_[i].end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
}

float
DPSolver::compute_score(int i, int j) {
  return context_->compute_score(i, j);
}

float
DPSolver::compute_ambient_score(float a, float b) {
  return context_->compute_ambient_score(a, b);
}


void
DPSolver::create_multiple_clustering_case() {
  // reset optimal_score_
  optimal_score_ = 0.;

  // sort vectors by priority function G(x,y) = x/y
  sort_by_priority(a_, b_);

  // create reference to score function
  if (parametric_dist_ == objective_fn::Gaussian) {
    context_ = std::make_unique<GaussianContext>(a_, 
						 b_, 
						 n_, 
						 parametric_dist_,
						 risk_partitioning_objective_,
						 use_rational_optimization_);
  }
  else if (parametric_dist_ == objective_fn::Poisson) {
    context_ = std::make_unique<PoissonContext>(a_, 
						b_, 
						n_,
						parametric_dist_,
						risk_partitioning_objective_,
						use_rational_optimization_);
  }
  else if (parametric_dist_ == objective_fn::RationalScore) {
    context_ = std::make_unique<RationalScoreContext>(a_, 
						      b_, 
						      n_,
						      parametric_dist_,
						      risk_partitioning_objective_,
						      use_rational_optimization_);
  }
  else {
    throw distributionException();
  }

  // Initialize LTSSSolver for t = 2 case
  LTSSSolver_ = std::make_unique<LTSSSolver>(n_, a_, b_, parametric_dist_);

  // Initialize matrix
  maxScore_ = std::vector<std::vector<float> >(n_, std::vector<float>(T_+1, std::numeric_limits<float>::min()));
  maxScore_sec_ = std::vector<std::vector<float> >(n_, std::vector<float>(T_+1, std::numeric_limits<float>::min()));
  nextStart_ = std::vector<std::vector<int> >(n_, std::vector<int>(T_+1, -1));
  nextStart_sec_ = std::vector<std::vector<int> >(n_, std::vector<int>(T_+1, -1));
  subsets_ = std::vector<std::vector<int> >(T_, std::vector<int>());
  score_by_subset_ = std::vector<float>(T_, 0.);

  // Fill in first,second columns corresponding to T = 0,1
  for(int j=0; j<2; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore_[i][j] = 0.;
      nextStart_[i][j] = (j==0)?-1:n_;
      maxScore_sec_[i][j] = (j==0)?0.:compute_score(i,n_);
      nextStart_sec_[i][j] = (j==0)?-1:n_;
    }
  }

  std::vector<float> a_atten, b_atten;
  for (int i=0; i<n_; ++i) {
    std::copy(a_.begin()+i, a_.end(), std::back_inserter(a_atten));
    std::copy(b_.begin()+i, b_.end(), std::back_inserter(b_atten));	      
    LTSSSolver_.reset(new LTSSSolver(n_-i, a_atten, b_atten, parametric_dist_));
    maxScore_[i][2] =  LTSSSolver_->get_optimal_score_extern();
    if (LTSSSolver_->get_optimal_subset_extern()[0] == 0) {
      int ind = LTSSSolver_->get_optimal_subset_extern().size()-1;
      nextStart_[i][2] = LTSSSolver_->get_optimal_subset_extern()[ind]+i+1;
    }
    else {
      nextStart_[i][2] = LTSSSolver_->get_optimal_subset_extern()[0]+i;
    }
    a_atten.clear(); b_atten.clear();
  }

  // Precompute partial sums
  std::vector<std::vector<float> > partialSums;
  partialSums = std::vector<std::vector<float> >(n_, std::vector<float>(n_, 0.));
  for (int i=0; i<n_; ++i) {
    for (int j=i; j<n_; ++j) {
      partialSums[i][j] = compute_score(i, j);
    }
  }

  // Fill in column-by-column from the left
  float score, score_sec;
  float maxScore, maxScore_sec;
  int maxNextStart = -1, maxNextStart_sec = -1;
  for(int j=2; j<=T_; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore = std::numeric_limits<float>::min();
      maxScore_sec = std::numeric_limits<float>::min();
      for (int k=i+1; k<=(n_-(j-1)); ++k) {
	score_sec = partialSums[i][k] + maxScore_sec_[k][j-1];
	score = std::max(partialSums[i][k] + maxScore_[k][j-1], maxScore_sec_[k][j-1]);
	if (score_sec > maxScore_sec) {
	  maxScore_sec = score_sec;
	  maxNextStart_sec = k;
	}
	if (score > maxScore) {
	  maxScore = score;
	  maxNextStart = k;
	}
      }
      if (j > 2) {
	maxScore_[i][j] = maxScore;
	nextStart_[i][j] = maxNextStart;
      }
      maxScore_sec_[i][j] = maxScore_sec;
      nextStart_sec_[i][j] = maxNextStart_sec;
      // Only need the initial entry in last column
      if (j == T_)
	break;
    }
  }  
}

void 
DPSolver::create() {
  // reset optimal_score_
  optimal_score_ = 0.;

  // sort vectors by priority function G(x,y) = x/y
  sort_by_priority(a_, b_);
  
  // create reference to score function

  if (parametric_dist_ == objective_fn::Gaussian) {
    context_ = std::make_unique<GaussianContext>(a_, 
						 b_, 
						 n_, 
						 parametric_dist_,
						 risk_partitioning_objective_,
						 use_rational_optimization_);
  }
  else if (parametric_dist_ == objective_fn::Poisson) {
    context_ = std::make_unique<PoissonContext>(a_, 
						b_, 
						n_,
						parametric_dist_,
						risk_partitioning_objective_,
						use_rational_optimization_);
  }
  else if (parametric_dist_ == objective_fn::RationalScore) {
    context_ = std::make_unique<RationalScoreContext>(a_, 
						      b_, 
						      n_,
						      parametric_dist_,
						      risk_partitioning_objective_,
						      use_rational_optimization_);
  }
  else {
    throw distributionException();
  }
  
  // Initialize matrix
  maxScore_ = std::vector<std::vector<float> >(n_, std::vector<float>(T_+1, std::numeric_limits<float>::min()));
  nextStart_ = std::vector<std::vector<int> >(n_, std::vector<int>(T_+1, -1));
  subsets_ = std::vector<std::vector<int> >(T_, std::vector<int>());
  score_by_subset_ = std::vector<float>(T_, 0.);

  // Fill in first,second columns corresponding to T = 0,1
  for(int j=0; j<2; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore_[i][j] = (j==0)?0.:compute_score(i,n_);
      nextStart_[i][j] = (j==0)?-1:n_;
    }
  }

  // Precompute partial sums
  std::vector<std::vector<float> > partialSums;
  partialSums = std::vector<std::vector<float> >(n_, std::vector<float>(n_, 0.));
  for (int i=0; i<n_; ++i) {
    for (int j=i; j<n_; ++j) {
      partialSums[i][j] = compute_score(i, j);
    }
  }

  // Fill in column-by-column from the left
  float score;
  float maxScore;
  int maxNextStart = -1;
  for(int j=2; j<=T_; ++j) {
    for (int i=0; i<n_; ++i) {
      maxScore = std::numeric_limits<float>::min();
      for (int k=i+1; k<=(n_-(j-1)); ++k) {
	score = partialSums[i][k] + maxScore_[k][j-1];
	if (score > maxScore) {
	  maxScore = score;
	  maxNextStart = k;
	}
      }
      maxScore_[i][j] = maxScore;
      nextStart_[i][j] = maxNextStart;
      // Only need the initial entry in last column
      if (j == T_)
	break;
    }
  }  
}

void
DPSolver::optimize_multiple_clustering_case() {
  // Pick out associated maxScores element
  int currentInd = 0, nextInd = 0, nextInd1 = 0;
  for (int t=T_; t>0; --t) {
    float score_num1 = 0., score_den1 = 0.;
    std::vector<int> subset;
    nextInd1 = nextStart_[currentInd][t];
    for (int i=currentInd; i<nextInd1; ++i) {
      score_num1 += a_[i];
      score_den1 += b_[i];
      subset.push_back(priority_sortind_[i]);
    }
    subsets_[T_-t] = subset;
    score_by_subset_[T_-t] = compute_ambient_score(score_num1, score_den1);
    nextInd = nextInd1;
    optimal_score_ += score_by_subset_[T_-t];
    currentInd = nextInd;
    
    // Early stopping, this could correspond to an optimal single subset being
    // the entire set at the LTSS (t = 2) stage
    if ((t > 1) && (currentInd == n_)) {
      break;
    }
  }

  // reorder subsets
  reorder_subsets(subsets_, score_by_subset_);

}

void
DPSolver::reorder_subsets(std::vector<std::vector<int> >& subsets, 
			  std::vector<float>& score_by_subsets) {
  std::vector<int> ind(subsets.size(), 0);
  std::iota(ind.begin(), ind.end(), 0.);

  std::stable_sort(ind.begin(), ind.end(),
		   [score_by_subsets](int i, int j) {
		     return (score_by_subsets[i] < score_by_subsets[j]);
		   });

  // Inefficient reordering
  std::vector<std::vector<int> > subsets_s;
  std::vector<float> score_by_subsets_s;
  subsets_s = std::vector<std::vector<int> >(subsets.size(), std::vector<int>());
  score_by_subsets_s = std::vector<float>(subsets.size(), 0.);

  for (size_t i=0; i<subsets.size(); ++i) {
    subsets_s[i] = subsets[ind[i]];
    score_by_subsets_s[i] = score_by_subsets[ind[i]];
  }

  std::copy(subsets_s.begin(), subsets_s.end(), subsets.begin());
  std::copy(score_by_subsets_s.begin(), score_by_subsets_s.end(), score_by_subsets.begin());
		   
  
}

void
DPSolver::optimize() {
  // Pick out associated maxScores element
  int currentInd = 0, nextInd = 0;
  for (int t=T_; t>0; --t) {
    float score_num = 0., score_den = 0.;
    nextInd = nextStart_[currentInd][t];
    for (int i=currentInd; i<nextInd; ++i) {
      subsets_[T_-t].push_back(priority_sortind_[i]);
      score_num += a_[priority_sortind_[i]];
      score_den += b_[priority_sortind_[i]];
    }
    score_by_subset_[T_-t] = compute_ambient_score(score_num, score_den);
    optimal_score_ += score_by_subset_[T_-t];
    currentInd = nextInd;
  }
}

std::vector<std::vector<int> >
DPSolver::get_optimal_subsets_extern() const {
  return subsets_;
}

float
DPSolver::get_optimal_score_extern() const {
  if (risk_partitioning_objective_) {
    return optimal_score_;
  }
  else {
    return std::accumulate(score_by_subset_.begin()+1, score_by_subset_.end(), 0.);
  }
}

std::vector<float>
DPSolver::get_score_by_subset_extern() const {
  return score_by_subset_;
}
