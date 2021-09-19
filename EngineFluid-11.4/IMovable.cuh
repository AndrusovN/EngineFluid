#ifndef I_MOVABLE
#define I_MOVABLE
#include "thrust/device_vector.h"

template <typename T>
class vector : public thrust::detail::vector_base<T, std::allocator<T>> {};

class IMovable {
public:
	__host__ virtual void moveToDevice() = 0;
	__host__ virtual void moveToHost() = 0;
};

#endif