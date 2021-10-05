#ifndef VECTOR_3
#define VECTOR_3
#include "cuda_runtime.h"

typedef float number_t;

#define NUMBER_T_INFTY 3e37
#define SQRT sqrtf
#define ATAN2 atan2f
#define EPSILON 0.000001f

class Vector3 {
private:
	number_t _x, _y, _z;
public:
	__host__ __device__ Vector3();
	__host__ __device__ Vector3(number_t x, number_t y, number_t z);

	__host__ __device__ ~Vector3();

	__host__ __device__ Vector3 operator = (const Vector3& other);

	__host__ __device__ number_t x() const;
	__host__ __device__ number_t y() const;
	__host__ __device__ number_t z() const;

	__host__ __device__ void set_x(number_t x);
	__host__ __device__ void set_y(number_t y);
	__host__ __device__ void set_z(number_t z);
	__host__ __device__ void normalize();

	__host__ __device__ number_t magnitude() const;
	__host__ __device__ number_t sqrMagnitude() const;
	__host__ __device__ number_t* toCArray() const;

	__host__ __device__ bool operator == (const Vector3& other) const;
	__host__ __device__ bool operator != (const Vector3& other) const;
	__host__ __device__ const Vector3 operator * (number_t value) const;
	__host__ __device__ const Vector3 operator / (number_t value) const;
	__host__ __device__ const Vector3 operator + (const Vector3& other) const;
	__host__ __device__ const Vector3 operator - (const Vector3& other) const;
	__host__ __device__ Vector3 operator *= (number_t other);
	__host__ __device__ Vector3 operator /= (number_t other);
	__host__ __device__ Vector3 operator += (const Vector3& other);
	__host__ __device__ Vector3 operator -= (const Vector3& other);

	__host__ __device__ const Vector3 normalized();

	__host__ __device__ number_t dot(const Vector3& other) const;
	__host__ __device__ const Vector3 cross(const Vector3& other) const;
	__host__ __device__ const Vector3 multiplyComponentWise(const Vector3& other) const;

	__host__ __device__ number_t angle(const Vector3& other) const;
	__host__ __device__ number_t unsignedAngle(const Vector3& other) const;
	__host__ __device__ number_t angle_sin(const Vector3& other) const;
	__host__ __device__ number_t angle_cos(const Vector3& other) const;

	__host__ __device__ const Vector3 reflect(const Vector3& planeNormal) const;

	__host__ __device__ static const Vector3 UP() {
		return Vector3(0, 1, 0);
	}

	__host__ __device__ static const Vector3 DOWN() {
		return Vector3(0, -1, 0);
	}

	__host__ __device__ static const Vector3 RIGHT() {
		return Vector3(1, 0, 0);
	}

	__host__ __device__ static const Vector3 LEFT() {
		return Vector3(-1, 0, 0);
	}

	__host__ __device__ static const Vector3 FORWARD() {
		return Vector3(0, 0, 1);
	}

	__host__ __device__ static const Vector3 BACK() {
		return Vector3(0, 0, -1);
	}

	__host__ __device__ static const Vector3 ZERO() {
		return Vector3(0, 0, 0);
	}

	__host__ __device__ static const Vector3 INFINITY_VECTOR() {
		return Vector3(NUMBER_T_INFTY, NUMBER_T_INFTY, NUMBER_T_INFTY);
	}

	__host__ __device__ static number_t distance(const Vector3& a, const Vector3& b);
	__host__ __device__ static number_t sqrDistance(const Vector3& a, const Vector3& b);

};

#endif
