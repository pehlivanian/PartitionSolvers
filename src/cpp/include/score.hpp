#ifndef __SCORE_HPP__
#define __SCORE_HPP__

#include <vector>
#include <list>
#include <limits>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <exception>
#include <immintrin.h>
#include <string.h>
#include <thread>


#define UNUSED(expr) do { (void)(expr); } while (0)

enum class objective_fn { Gaussian = 0, 
			    Poisson = 1, 
			    RationalScore = 2 };

struct optimizationFlagException : public std::exception {
  const char* what() const throw () {
    return "Optimized version not implemented";
  };
};


namespace Objectives {
  template<typename DataType>
  class ParametricContext {

  public:

    ParametricContext(const std::vector<DataType>& a, 
		      const std::vector<DataType>& b, 
		      std::size_t n, 
		      bool risk_partitioning_objective,
		      bool use_rational_optimization,
		      std::string name
		      ) :
      a_{a},
      b_{b},
      n_{n},
      partialSums_{std::vector<std::vector<DataType>>(n+1, std::vector<DataType>(n+1, 0.))},
      risk_partitioning_objective_{risk_partitioning_objective},
      use_rational_optimization_{use_rational_optimization},
      // cache_{CacheType(n_+1, std::vector<int>(n_+1))},
      name_{name}

    {	
      if (false) {
	cache_ = (int**)malloc(sizeof(int*)*(n_+1));
	for (int i=0; i<n_+1; ++i) {
	  cache_[i] = (int*)malloc(sizeof(int)*(n_+1));
	  memset(cache_[i], -1, (n_+1)*sizeof(int));
	}
      }
    }

    ParametricContext() = default;
    virtual ~ParametricContext() = default;

    void init();

    DataType get_score(int, int) const;
    DataType get_ambient_score(DataType, DataType) const;
    std::vector<std::vector<DataType>> get_scores() const;

    std::string getName() const;
    bool getRiskPartitioningObjective() const;
    bool getUseRationalOptimization() const;

    std::vector<std::vector<DataType>> get_partial_sums_a() const;
    std::vector<std::vector<DataType>> get_partial_sums_b() const;

    // Only for testing/benchmark purposes
    // __compute_partial_sums__* fills a_sums_, b_sums_
    void __compute_partial_sums__() { compute_partial_sums(); }
    void __compute_partial_sums_AVX256__() { compute_partial_sums_AVX256(); }
    void __compute_partial_sums_parallel__() { compute_partial_sums_parallel(); }
    
    // __compute_scores__ fills a_sums_, b_sums_, partialSums_
    // Note that partialSums_ is the precached scores by (i, j)
    void __compute_scores__() { compute_scores(); }
    void __compute_scores_parallel__() { compute_scores_parallel(); }
    void __compute_scores_AVX256__() { compute_scores_AVX256(); }

    // __compute_score__ computes the score by (i, j)
    // based on precached a_sums_, b_sums_
    DataType __compute_score__(int i, int j) { return compute_score(i, j); }
    DataType __compute_ambient_score__(DataType a, DataType b) { return compute_ambient_score(a, b); }

  protected:
    virtual DataType compute_score_multclust(int, int)		         = 0;
    virtual DataType compute_score_riskpart(int, int)		         = 0;
    virtual DataType compute_ambient_score_multclust(DataType, DataType) = 0;
    virtual DataType compute_ambient_score_riskpart(DataType, DataType)  = 0;

    virtual DataType compute_score_multclust_optimized(int, int)         = 0;
    virtual DataType compute_score_riskpart_optimized(int, int)		 = 0;

    DataType compute_score_multclust_memoized(int, int);
    DataType compute_score_riskpart_memoized(int, int);

    void compute_partial_sums();
    void compute_partial_sums_AVX256();
    void compute_partial_sums_parallel();
    virtual void compute_scores();
    virtual void compute_scores_AVX256();
    virtual void compute_scores_parallel();

    // DataType inline compute_score(int, int);
    // DataType inline compute_ambient_score(DataType, DataType);
    DataType compute_score(int, int);
    DataType compute_ambient_score(DataType, DataType);

    std::vector<DataType> a_;
    std::vector<DataType> b_;
    std::size_t n_;
    std::vector<std::vector<DataType> > a_sums_;
    std::vector<std::vector<DataType> > b_sums_;
    std::vector<std::vector<DataType>> partialSums_;
    bool risk_partitioning_objective_;
    bool use_rational_optimization_;
    std::string name_;

    int **cache_;

  };

  
  template<typename DataType>
  class PoissonContext : public ParametricContext<DataType> {

  public:
    PoissonContext(const std::vector<DataType>& a, 
		   const std::vector<DataType>& b, 
		   std::size_t n, 
		   bool risk_partitioning_objective,
		   bool use_rational_optimization) : ParametricContext<DataType>{a,
		      b,
		      n,
		      risk_partitioning_objective,
		      use_rational_optimization,
		      "Poisson"}
    {}

    PoissonContext() = default;

  private:  
    DataType compute_score_multclust(int, int) override;
    DataType compute_score_riskpart(int, int) override;
    DataType compute_ambient_score_multclust(DataType, DataType) override;
    DataType compute_ambient_score_riskpart(DataType, DataType) override;
    DataType compute_score_riskpart_optimized(int, int) override;
    DataType compute_score_multclust_optimized(int, int) override;

  };

  template<typename DataType>
  class GaussianContext : public ParametricContext<DataType> {
  public:
    GaussianContext(const std::vector<DataType>& a, 
		    const std::vector<DataType>& b, 
		    std::size_t n, 
		    bool risk_partitioning_objective,
		    bool use_rational_optimization) : ParametricContext<DataType>(a,
										  b,
										  n,
										  risk_partitioning_objective,
										  use_rational_optimization,
										  "Gaussian")
    {}

    GaussianContext() = default;

  private:  
    DataType compute_score_multclust(int, int) override;
    DataType compute_score_riskpart(int, int) override;
    DataType compute_ambient_score_multclust(DataType, DataType) override;
    DataType compute_ambient_score_riskpart(DataType, DataType) override;
    DataType compute_score_multclust_optimized(int, int) override;
    DataType compute_score_riskpart_optimized(int, int) override;

  };

  template<typename DataType>
  class RationalScoreContext : public ParametricContext<DataType> {
    // This class doesn't correspond to any regular exponential family,
    // it is used to define ambient functions on the partition polytope
    // for targeted applications - quadratic approximations to loss, for
    // XGBoost, e.g.
  public:
    RationalScoreContext(const std::vector<DataType>& a,
			 const std::vector<DataType>& b,
			 std::size_t n,
			 bool risk_partitioning_objective,
			 bool use_rational_optimization) : ParametricContext<DataType>(a,
										       b,
										       n,
										       risk_partitioning_objective,
										       use_rational_optimization,
										       "RationalScore")
    {}

    RationalScoreContext() = default;

  private:
    DataType compute_score_multclust(int, int) override;
    DataType compute_score_riskpart(int, int) override;
    DataType compute_ambient_score_multclust(DataType, DataType) override;
    DataType compute_ambient_score_riskpart(DataType, DataType) override;
    DataType compute_score_riskpart_optimized(int, int) override;
    DataType compute_score_multclust_optimized(int, int) override;

  };

}
#include "score_impl.hpp"


#endif
