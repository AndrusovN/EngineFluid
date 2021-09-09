#ifndef QUATERNION
#define QUATERNION

#include "Vector3.cuh"

class Quaternion {
private:
	number_t _x, _y, _z, _w;

public:
	Quaternion(number_t x, number_t y, number_t z, number_t w);
	Quaternion();
	Quaternion(const Vector3& base);

	__host__ __device__ const Quaternion operator * (number_t value) const;
	__host__ __device__ const Quaternion operator / (number_t value) const;

	__host__ __device__ const Quaternion operator + (const Quaternion& other) const;
	__host__ __device__ const Quaternion operator - (const Quaternion& other) const;

	__host__ __device__ const Quaternion operator * (const Quaternion& other) const;
	__host__ __device__ const bool operator == (const Quaternion& other) const;
	__host__ __device__ const bool operator != (const Quaternion& other) const;
	__host__ __device__ const Quaternion conjugated() const;
	__host__ __device__ const Quaternion inversed() const;

	__host__ __device__ const number_t magnitude() const;
	__host__ __device__ const number_t sqrMagnitude() const;

	__host__ __device__ const Vector3 toVector3() const;
	__host__ __device__ const Vector3 applyToVector(Vector3 vector) const;

	__host__ __device__ static Quaternion fromAngle(number_t angle, Vector3 axis);

	static const Quaternion IDENTITY;
};

#endif
