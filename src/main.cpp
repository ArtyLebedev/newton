#include <iostream>
#include <cmath>
#include <fstream>

#include "mpi_solver/gauss_solver.h"
#include "ntn_comp.h"
#include "common.h"

namespace ntn {
	size_t system_size = 1024;
	vec sum(const vec& lhs, const vec& rhs) {
		vec res(lhs.size());
		if(lhs.size() != rhs.size())
			return res;

		for(size_t i = 0; i < lhs.size(); ++i) {
			res[i] = lhs[i] + rhs[i];
		}
		return res;
	}

	double diff(const vec& lhs, const vec& rhs) {
		double res = 0.;
		for(size_t i = 0; i < lhs.size(); ++i) {
			res += lhs[i] - rhs[i];
		}
		return fabs(res);
	}

	double func(const size_t& i, const vec& v) {
		double res = 0.;
		res = cos(v[i]) - 1;
		return res;
	}

	double derivative(const size_t& i, const size_t& j, const vec& v) {
		double res = 0.;
		if(i == j){
			res = -sin(v[i]);
		}
		return res;
	}
}

int main(int argc, char** argv) {
	ntn::system_size = atoi(argv[1]);
	int rank, size;
	ntn::system s;
	ntn::derivatives d(ntn::system_size);
	ntn::vec start(ntn::system_size, 0.87);

	for(size_t i = 0; i < ntn::system_size; ++i) {
		s.push_back(&ntn::func);
	}
	for(size_t i = 0; i < ntn::system_size; ++i) {
		for(size_t j = 0; j < ntn::system_size; ++j)
			d[i].push_back(&ntn::derivative);
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	ntn::newtone n(rank);

	MPI_Barrier(MPI_COMM_WORLD);
	double time = MPI_Wtime();
	ntn::vec sol = n.find_solution(s, start, d, &ntn::gauss_solver, &ntn::sum, &ntn::diff, 0.0001, 100);
	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime() - time;
	double max_time;
	MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		std::ofstream myfile;
		char filename[32];
		snprintf(filename, 32, "out_%ld_%d.txt", ntn::system_size, size);
		myfile.open (filename);
		for(size_t i = 0; i < sol.size(); ++i) {
			myfile << sol[i] << " ";
		}
		myfile << std::endl;
		myfile << "Time: " << time << " " << max_time << std::endl;
		myfile.close();
	}
	MPI_Finalize();
	return 0;
}


