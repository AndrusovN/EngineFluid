#ifndef I_MOVABLE
#define I_MOVABLE

#define TYPE_IMOVABLE_CHECKER_CONSTANT 179

class IMovable {
public:
	unsigned char type_checker;
	__host__ __device__ IMovable() {
		type_checker = TYPE_IMOVABLE_CHECKER_CONSTANT;
	}

	__host__ virtual void moveToDevice() = 0;
	__host__ virtual void moveToHost() = 0;
};

__host__ __device__ bool isMovable(void* object) {
	char checker = ((IMovable*)object)->type_checker;
	return checker == TYPE_IMOVABLE_CHECKER_CONSTANT;
}

#endif