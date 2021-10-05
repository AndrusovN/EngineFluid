#ifndef I_COPYABLE
#define I_COPYABLE
#include "cuda_runtime.h"

#define ICPYABLE_TYPE_CHECKER_VALUE 121

class ICopyable {
public:
	unsigned char type_checker;
	__host__ __device__ ICopyable() {
		type_checker = ICPYABLE_TYPE_CHECKER_VALUE;
	}
	__host__ __device__ virtual ICopyable* copy() = 0;
};

__host__ __device__ bool isCopyable(void* object) {
	ICopyable* obj = (ICopyable*)object;
	return obj->type_checker == ICPYABLE_TYPE_CHECKER_VALUE;
}

#endif