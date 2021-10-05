#ifndef PAIR
#define PAIR

template <typename T1, typename T2>
class Pair {
public:
	T1 first;
	T2 second;
	__host__ __device__ Pair() {}
	__host__ __device__ Pair(T1 first, T2 second) {
		this->first = first;
		this->second = second;
	}
};


#endif