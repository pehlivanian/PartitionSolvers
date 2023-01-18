#include "score.hpp"

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
