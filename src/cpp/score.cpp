#include "score.hpp"

const int NUMTHREADS = 10;

void 
Objectives::ParametricContext::compute_partial_sums() {
  a_sums_ = std::vector<std::vector<float> >(n_, std::vector<float>(n_+1, std::numeric_limits<float>::lowest()));
  b_sums_ = std::vector<std::vector<float> >(n_, std::vector<float>(n_+1, std::numeric_limits<float>::lowest()));
  
  for (int i=0; i<n_; ++i) {
    a_sums_[i][i] = 0.;
    b_sums_[i][i] = 0.;
  }
  
  for (int i=0; i<n_; ++i) {
    for (int j=i+1; j<=n_; ++j) {
      a_sums_[i][j] = a_sums_[i][j-1] + a_[j-1];
      b_sums_[i][j] = b_sums_[i][j-1] + b_[j-1];
    }
  }  
}

void
Objectives::ParametricContext::compute_partial_sums_AVX256() {

  a_sums_ = std::vector<std::vector<float>>(n_+1, std::vector<float>(n_, std::numeric_limits<float>::lowest()));
  b_sums_ = std::vector<std::vector<float>>(n_+1, std::vector<float>(n_, std::numeric_limits<float>::lowest()));

  for (int j=0; j<n_; ++j) {
    a_sums_[j][j] = 0.;
    b_sums_[j][j] = 0.;
  }

  float r_[8];
  for (int j=1; j<n_+1; ++j) {
    int unroll = (j/4)*4, i=0;
    for (; i<unroll; i+=4) {
      __m256 v1 = _mm256_set_ps(a_sums_[j-1][i], a_sums_[j-1][i+1], a_sums_[j-1][i+2], a_sums_[j-1][i+3],
				b_sums_[j-1][i], b_sums_[j-1][i+1], b_sums_[j-1][i+2], b_sums_[j-1][i+3]); 
      __m256 v2 = _mm256_set_ps(a_[j-1], a_[j-1], a_[j-1], a_[j-1],
				b_[j-1], b_[j-1], b_[j-1], b_[j-1]);
      __m256 r = _mm256_add_ps(v1, v2);
      memcpy(r_, &r, sizeof(r_));
      a_sums_[j][i]   = r_[7];
      a_sums_[j][i+1] = r_[6];
      a_sums_[j][i+2] = r_[5];
      a_sums_[j][i+3] = r_[4];
      b_sums_[j][i]   = r_[3];
      b_sums_[j][i+1] = r_[2];
      b_sums_[j][i+2] = r_[1];
      b_sums_[j][i+3] = r_[0];
    }
    
    for(;i<j;++i) {
      a_sums_[j][i] = a_sums_[j-1][i] + a_[j-1];
      b_sums_[j][i] = b_sums_[j-1][i] + b_[j-1];      
    }
  }
}

void
Objectives::ParametricContext::compute_scores_parallel(std::vector<std::vector<float>>& partialSums) {

  const int numThreads = NUMTHREADS;

  a_sums_ = std::vector<std::vector<float> >(n_, std::vector<float>(n_+1, std::numeric_limits<float>::lowest()));
  b_sums_ = std::vector<std::vector<float> >(n_, std::vector<float>(n_+1, std::numeric_limits<float>::lowest()));
  partialSums = std::vector<std::vector<float>>(n_, std::vector<float>(n_+1, 0.));
  
  for (int i=0; i<n_; ++i) {
    a_sums_[i][i] = 0.;
    b_sums_[i][i] = 0.;
  }

  auto task_ab_block = [this, &partialSums](int ind1, int ind2) {
    for (int i=ind1; i<ind2; ++i) {
      for (int j=i+1; j<=this->n_; ++j) {
	this->a_sums_[i][j] = this->a_sums_[i][j-1] + this->a_[j-1];
	this->b_sums_[i][j] = this->b_sums_[i][j-1] + this->b_[j-1];
	partialSums[i][j] = this->compute_score(i, j);
      }
    }
  };

  int blockSize = static_cast<float>(n_)/static_cast<float>(numThreads);
  int startOfBlock = 0, endOfBlock = startOfBlock + blockSize;
  
  std::vector<std::thread> threads;

  while (endOfBlock < n_) {
    threads.emplace_back(task_ab_block, startOfBlock, endOfBlock);
    startOfBlock = endOfBlock; endOfBlock+= blockSize;
   }
  threads.emplace_back(task_ab_block, startOfBlock, n_);
  
  for (auto it=threads.begin(); it!=threads.end(); ++it)
    it->join();

}

void
Objectives::ParametricContext::compute_partial_sums_parallel() {

  const int numThreads = NUMTHREADS;

  a_sums_ = std::vector<std::vector<float> >(n_, std::vector<float>(n_+1, std::numeric_limits<float>::lowest()));
  b_sums_ = std::vector<std::vector<float> >(n_, std::vector<float>(n_+1, std::numeric_limits<float>::lowest()));

  for (int i=0; i<n_; ++i) {
    a_sums_[i][i] = 0.;
    b_sums_[i][i] = 0.;
  }

  auto task_ab_block = [this](int ind1, int ind2) {
    for (int i=ind1; i<ind2; ++i) {
      for (int j=i+1; j<=this->n_; ++j) {
	this->a_sums_[i][j] = this->a_sums_[i][j-1] + this->a_[j-1];
	this->b_sums_[i][j] = this->b_sums_[i][j-1] + this->b_[j-1];
      }
    }
  };

  int blockSize = static_cast<float>(n_)/static_cast<float>(numThreads);
  int startOfBlock = 0, endOfBlock = startOfBlock + blockSize;
  
  std::vector<std::thread> threads;

  while (endOfBlock < n_) {
    threads.emplace_back(task_ab_block, startOfBlock, endOfBlock);
    startOfBlock = endOfBlock; endOfBlock+= blockSize;
   }
  threads.emplace_back(task_ab_block, startOfBlock, n_);
  
  for (auto it=threads.begin(); it!=threads.end(); ++it)
    it->join();

}


float
Objectives::ParametricContext::compute_score(int i, int j) {
  if (risk_partitioning_objective_) {
    if (use_rational_optimization_) {
      return compute_score_riskpart_optimized(i, j);
    }
    else {
      return compute_score_riskpart(i, j);
    }
  }
  else {
    if (use_rational_optimization_) {
      return compute_score_multclust_optimized(i, j);
    }
    else {
      return compute_score_multclust(i, j);
    }
  }
}

float 
Objectives::ParametricContext::compute_ambient_score(float a, float b) {
  if (risk_partitioning_objective_) {
    return compute_ambient_score_riskpart(a, b);
  }
  else {
    return compute_ambient_score_multclust(a, b);
  }
}
