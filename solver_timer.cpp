#include <stdexcept>

#include "solver_timer.hpp"

#define DEBUG true

using namespace Objectives;

auto main(int argc, char **argv) -> int {

  int n, T, NStride, TStride;

  // args:
  // (n, T, NStride, TStride)

  if (argc < 5)
    throw std::invalid_argument("Must call with 4 input arguments.");

  std::istringstream nss(argv[1]), Tss(argv[2]), NStridess(argv[3]), TStridess(argv[4]);
  nss >> n; Tss >> T; NStridess >> NStride; TStridess >> TStride;

  //
  // Fixed parameters
  //
  // Settable: ambient distribution assumption, risk_partitioning/multiple_clustering mode,
  // optimized. Optimized should always be turned on; it uses a lookup table of order n^2 to
  // gain much runtime efficiency.
  //
  auto dist = objective_fn::Poisson;
  bool risk_partitioning = true;
  bool optimized = true;
  //
  //

  float a_lb = -10., a_ub = 10., b_lb = .00001, b_ub = 10.;
  if (dist == objective_fn::Poisson) {
    a_lb = b_lb;
  }

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dista(a_lb, a_ub);
  std::uniform_real_distribution<double> distb(b_lb, b_ub);

  auto gena = [&dista, &mersenne_engine](){ return dista(mersenne_engine); };
  auto genb = [&distb, &mersenne_engine](){ return distb(mersenne_engine); };

  using dur = std::chrono::high_resolution_clock::duration::rep;
  std::vector<std::vector<dur> > times(n+1);
  for (int i=0; i<=n; ++i)
    {
      times[i] = std::vector<dur>(T+1);
      for(int j=0; j<=T; ++j)
	times[i][j] = dur{0};
    }

  // ground set size ranges from T, T+NStride, ..., n
  for (int sampleSize=T; sampleSize<=n; sampleSize+=NStride) { 
    // size of partition ranges from TStride, 2*TStride, ..., T
    for (int numParts=TStride; numParts<=T; numParts+=TStride) {


  // for (int sampleSize=n; sampleSize<=n; sampleSize+=stride) {
  // for (int numParts=10;numParts<=T;numParts+=partStride) {

      if (sampleSize > numParts) {

	std::vector<float> a(sampleSize), b(sampleSize);
	
	std::generate(a.begin(), a.end(), gena);
	std::generate(b.begin(), b.end(), genb);
	
	//======================
	//= Start timing block =
	//======================
	precise_timer timer;
	auto dp = DPSolver(sampleSize, numParts, a, b, dist, risk_partitioning, optimized);
	auto et = timer.elapsed_time<unsigned int, std::chrono::microseconds>();
	//====================
	//= End timing block =
	//====================

	if (DEBUG) {
	  if (!(sampleSize%100)) {
	    std::cerr << "Completed case (n,T) = (" << sampleSize << ", " << numParts << "): " 
		      << et
		      << " microseconds"
		      << std::endl;
	  }
	}
	times[sampleSize][numParts] = et;
      }

    }
  }
  
  // dump
  auto delim = ",";
  for( const auto &v : times) {
    for( const auto &el : v)
      std::cout << el << delim;
    std::cout << std::endl;
  }
  
  return 0;
}
