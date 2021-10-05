#ifndef QUATERNION
#define QUATERNION

#include "Vector3.cuh"

__host__ __device__ number_t PI() { return 3.14159265358979323846264338327950; }

class Quaternion {
private:
	number_t _x, _y, _z, _w;

public:
	__host__ __device__ Quaternion(number_t x, number_t y, number_t z, number_t w);
	__host__ __device__ Quaternion();
	__host__ __device__ Quaternion(const Vector3& base);

	__host__ __device__ const Quaternion operator * (number_t value) const;
	__host__ __device__ const Quaternion operator / (number_t value) const;
	__host__ __device__ Quaternion operator *= (number_t value);
	__host__ __device__ Quaternion operator /= (number_t value);

	__host__ __device__ const Quaternion operator + (const Quaternion& other) const;
	__host__ __device__ const Quaternion operator - (const Quaternion& other) const;
	__host__ __device__ Quaternion operator += (const Quaternion& other);
	__host__ __device__ Quaternion operator -= (const Quaternion& other);

	__host__ __device__ const Quaternion operator * (const Quaternion& other) const;
	__host__ __device__ const Quaternion operator / (const Quaternion& other) const;
	__host__ __device__ Quaternion operator *= (const Quaternion& other);
	__host__ __device__ Quaternion operator /= (const Quaternion& other);

	__host__ __device__ bool operator == (const Quaternion& other) const;
	__host__ __device__ bool operator != (const Quaternion& other) const;
	__host__ __device__ const Quaternion conjugated() const;
	__host__ __device__ const Quaternion inversed() const;

	__host__ __device__ number_t magnitude() const;
	__host__ __device__ number_t sqrMagnitude() const;

	__host__ __device__ const Vector3 toVector3() const;
	__host__ __device__ const Vector3 applyToVector(Vector3 vector) const;

	__host__ __device__ static Quaternion fromAngle(number_t angle, Vector3 axis);

	__host__ __device__ static const Quaternion IDENTITY() {
		return Quaternion(1, 0, 0, 0);
	}

};

#endif
