#ifndef SRC_COMMON_H_
#define SRC_COMMON_H_

#include <vector>
#include <cstdlib>
#include <limits>
#include <iostream>
#include "mpi.h"

namespace ntn {
	typedef std::vector<double> vec;
	typedef double (*function)(const size_t&, const vec&);
	typedef double (*deriv)(const size_t&, const size_t&, const vec&);
	typedef std::vector<function> system;
	typedef std::vector<std::vector<deriv> > derivatives;
	typedef std::vector<std::vector<double> > matrix;
	struct linear_system {
		matrix A;
		vec b;
	};
	typedef vec (*linear_system_solver)(const matrix&, const vec&);
	typedef vec (*vec_sum)(const vec&, const vec&);
	typedef double (*vec_diff)(const vec&, const vec&);

	static void print_vec(const vec& v) {
		std::cout << "size: " << v.size() << std::endl;
		for(size_t i = 0; i < v.size(); ++i) {
			std::cout << v[i] << " ";
		}
		std::cout << std::endl;
	}
}

#endif /* SRC_COMMON_H_ */
