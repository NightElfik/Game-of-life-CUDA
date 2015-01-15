#pragma once
#include <iostream>
#include <sstream>

namespace mf {

	#define checkCudaErrors(val)    checkCudaResult((val), #val, __FILE__, __LINE__)

	/// Checks result of CUDA operation and prints a message when an error occurs.
	/// This function should be used whenever the result of CUDA operation would be ignored otherwise.
	template<typename T>
	bool checkCudaResult(T result, char const *const func, const char *const file, int const line) {
		if (result) {
			if (result == cudaErrorCudartUnloading) {
				// Do not try to print error when program is shutting down.
				return false;
			}

			std::stringstream ss;
			ss << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
				<< " \"" << func << "\"";
			std::cerr << ss.str() << std::endl;
			return false;
		}
		return true;
	}

}