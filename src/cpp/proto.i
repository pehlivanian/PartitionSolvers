/* File : proto.i */
%module proto

%{
#include "DP.hpp"
#include "python_dpsolver.hpp"
#include "LTSS.hpp"
#include "python_ltsssolver.hpp"
%}

%include "std_vector.i"
%include "std_pair.i"

namespace std {
%template(IArray) vector<int>;
%template(FArray) vector<float>;
%template(IArrayArray) vector<vector<int> >;
%template(IArrayFPair) pair<vector<int>, float>;
%template(IArrayArrayFPair) pair<vector<vector<int> >, float>;
%template(SWCont) vector<pair<vector<vector<int> >, float> >;
%template(OLSPair) pair<vector<vector<int> >, int >;
%template(ScoreCont) pair<vector<pair<vector<vector<int> >,float> >, int>;
}

%include "python_dpsolver.hpp"
%include "python_ltsssolver.hpp"

