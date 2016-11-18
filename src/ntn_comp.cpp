#include "ntn_comp.h"

#include "mpi.h"

#include <iostream>

namespace ntn {
	vec newtone::find_solution(const system& sys, const vec& start, const derivatives& d,
									 linear_system_solver solver, vec_sum vec_summator,
									 vec_diff vec_differ, const double& eps,
									 const size_t& max_iter) {
		size_t iter_count = 1;
		double diff = 0.;
		vec sys_val(sys.size(), 0);
		if(m_rank == 0) {
			m_jac.reserve(sys.size());
			for(size_t i = 0; i < sys.size(); ++i) {
				vec v(sys.size());
				m_jac.push_back(v);
			}

			compute_jacobian(sys, d, start);

			for(size_t i = 0; i < sys.size(); ++i) {
				sys_val[i] = -sys[i](i, start);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		vec delta = solver(m_jac, sys_val);
		vec new_sol(sys.size()), old_sol(sys.size());
		if(m_rank == 0) {
			new_sol = vec_summator(start, delta);
			old_sol = start;
		}

		MPI_Bcast(new_sol.data(), new_sol.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(old_sol.data(), old_sol.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		diff = vec_differ(new_sol, old_sol);

		while(diff > eps && iter_count <= max_iter) {
			old_sol = new_sol;
			if(m_rank == 0) {
				compute_jacobian(sys, d, old_sol);
				for(size_t i = 0; i < sys.size(); ++i) {
					sys_val[i] = -sys[i](i, old_sol);
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			delta = solver(m_jac, sys_val);
			if(m_rank == 0) {
				new_sol = vec_summator(old_sol, delta);
			}
			MPI_Bcast(new_sol.data(), new_sol.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
			iter_count++;
			diff = vec_differ(new_sol, old_sol);
		}

		return new_sol;
	}

	double newtone::compute_derivative(const size_t& pos, function func, const size_t& var_num, const vec& point) {
		vec left_point(point), right_point(point);
		left_point[var_num] -= m_h;
		right_point[var_num] += m_h;

		double left = func(pos, left_point), right = func(pos, right_point);

		return (right - left) / (2 * m_h);
	}

	void newtone::compute_jacobian(const system& sys, const derivatives& d, const vec& point) {
		for(size_t i = 0; i < sys.size(); ++i) {
			for(size_t j = 0; j < sys.size(); ++j) {
				double res_val;

				res_val = d[i][j](i, j, point);
				m_jac[i][j] = res_val;
			}
		}
	}
}




