#ifndef MESH
#define MESH
#include "Triangle.cuh"
#include "Quaternion.cuh"

class Mesh {
private:
	Triangle* _triangles;
	int _triangles_size = 0;
	Vector3 _center = Vector3::ZERO;
	number_t _scale = 1;
	Quaternion _appliedRotation;

	__host__ __device__ Vector3 rotatePoint(Vector3 point, Quaternion rotation);
public:
	__host__ __device__ Mesh(Vector3 position = Vector3::ZERO, number_t scale = 1);
	__host__ __device__ Mesh(Triangle* triangles, int size, Vector3 position = Vector3::ZERO, number_t scale = 1);
	__host__ Mesh(const char* filename, int stringSize, Vector3 position = Vector3::ZERO, number_t scale = 1);

	__host__ void moveToCUDA();

	__host__ __device__ Triangle get_triangle(int index);

	__host__ __device__ int size();

	__host__ __device__ void setAppliedRotation(Quaternion rotation);

	__host__ __device__ void rotate(Quaternion rotation);

	__host__ __device__ void translate(Vector3 offset);

	__host__ __device__ void setPosition(Vector3 position);

	__host__ __device__ bool isPointInside(Vector3 point);

	__host__ __device__ void setScale(number_t scale);

	__host__ __device__ void scale(number_t times);

	__host__ __device__ number_t getScale();
};

#endif