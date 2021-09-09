#ifndef TRIANGLE
#define TRIANGLE 

#include "Vector3.cuh"

// Triangle is immutable
class Triangle {
private:
	const Vector3 _vertices[3];
	const Vector3 _center, _normal;

	__host__ __device__ const bool isInsideTriangle(Vector3 point) const;
public:
	__host__ __device__ Triangle(Vector3 A, Vector3 B, Vector3 C);

	__host__ __device__ const Vector3 rayIntersection(const Vector3& startPoint, const Vector3& direction) const;
	__host__ __device__ const bool isInside(const Vector3& other) const;
	__host__ __device__ const Vector3 normal() const;
	__host__ __device__ const Triangle reversed() const;
	__host__ __device__ const Vector3 get_vertex(int index) const;
};

#endif