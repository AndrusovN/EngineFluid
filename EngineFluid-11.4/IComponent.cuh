#ifndef ICOMPONENT
#define ICOMPONENT

#include "Quaternion.cuh"

class IComponent {
public:
	__host__ __device__ virtual int typeId() = 0;

	__host__ __device__ virtual void rotate(Quaternion rotation) = 0;
	__host__ __device__ virtual void translate(Vector3 offset) = 0;

	__host__ __device__ virtual void update() = 0;
	__host__ __device__ virtual void start() = 0;
	__host__ __device__ virtual void awake() = 0;
	__host__ __device__ virtual void lateUpdate() = 0;

	__host__ __device__ virtual void onDestroy() = 0;
};

#endif