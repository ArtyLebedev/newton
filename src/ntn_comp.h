#ifndef NTN_COMP_H_
#define NTN_COMP_H_
#include <vector>
#include <cstdlib>
#include <limits>

#include "common.h"

namespace ntn {
	class newtone {
	public:
		newtone(int rank): m_rank(rank){m_h = std::numeric_limits<double>::epsilon();};
		vec find_solution(const system& sys, const vec& start, const derivatives& d,
								 linear_system_solver solver, vec_sum vec_summator,
								 vec_diff vec_differ, const double& eps,
								 const size_t& max_iter);
		double compute_derivative(const size_t& pos, function func, const size_t& var_num, const vec& point);
	private:
		void compute_jacobian(const system& sys, const derivatives& d, const vec& point);
		double m_h;
		int m_rank;
		matrix m_jac;
		vec  m_right_part;
	};
}




#endif /* NTN_COMP_H_ */
