#include "Quaternion.cuh"
#include "assert.h"

__host__ __device__ Quaternion::Quaternion(number_t x, number_t y, number_t z, number_t w) : _x(x), _y(y), _z(z), _w(w) {}

__host__ __device__ Quaternion::Quaternion() : _x(0), _y(0), _z(0), _w(0) {}

__host__ __device__ Quaternion::Quaternion(const Vector3& base) : _x(0), _y(base.x()), _z(base.y()), _w(base.z()) {}

__host__ __device__ const Quaternion Quaternion::operator*(number_t value) const
{
	return Quaternion(_x * value, _y * value, _z * value, _w * value);
}

__host__ __device__ const Quaternion Quaternion::operator/(number_t value) const
{
	assert(value > EPSILON);
	return *this * ((number_t)1 / value);
}

__host__ __device__ Quaternion Quaternion::operator*=(number_t value)
{
	return *this = *this * value;
}

__host__ __device__ Quaternion Quaternion::operator/=(number_t value)
{
	return *this = *this / value;
}

__host__ __device__ const Quaternion Quaternion::operator+(const Quaternion& other) const
{
	return Quaternion(_x + other._x, _y + other._y, _z + other._z, _w + other._w);
}

__host__ __device__ const Quaternion Quaternion::operator-(const Quaternion& other) const
{
	return *this + (other * -1);
}

__host__ __device__ Quaternion Quaternion::operator+=(const Quaternion& other)
{
	return *this = *this + other;
}

__host__ __device__ Quaternion Quaternion::operator-=(const Quaternion& other)
{
	return *this = *this - other;
}

__host__ __device__ const Quaternion Quaternion::operator*(const Quaternion& other) const
{
	number_t x = _x * other._x 
				- _y * other._y 
				- _z * other._z 
				- _w * other._w;

	number_t y = _x * other._y 
				+ _y * other._x 
				+ _z * other._w 
				- _w * other._z;

	number_t z = _x * other._z 
				+ _z * other._x 
				+ _w * other._y 
				- _y * other._w;

	number_t w = _x * other._w 
				+ _w * other._x 
				+ _y * other._z 
				- _z * other._y;

	return Quaternion(x, y, z, w);
}

__host__ __device__ const Quaternion Quaternion::operator/(const Quaternion& other) const
{
	return *this * other.inversed();
}

__host__ __device__ Quaternion Quaternion::operator*=(const Quaternion& other)
{
	return *this = *this * other;
}

__host__ __device__ Quaternion Quaternion::operator/=(const Quaternion& other)
{
	return *this = *this / other;
}

__host__ __device__ bool Quaternion::operator==(const Quaternion& other) const
{
	return _x == other._x &&
		_y == other._y &&
		_z == other._z &&
		_w == other._w;
}

__host__ __device__ bool Quaternion::operator!=(const Quaternion& other) const
{
	return !(*this == other);
}

__host__ __device__ const Quaternion Quaternion::conjugated() const
{
	return Quaternion(_x, -_y, -_z, -_w);
}

__host__ __device__ const Quaternion Quaternion::inversed() const
{
	return conjugated() / sqrMagnitude();
}

__host__ __device__ number_t Quaternion::magnitude() const
{
	return SQRT(sqrMagnitude());
}

__host__ __device__ number_t Quaternion::sqrMagnitude() const
{
	return (*this * conjugated())._x;
}

__host__ __device__ const Vector3 Quaternion::toVector3() const
{
	return Vector3(_y, _z, _w);
}

__host__ __device__ const Vector3 Quaternion::applyToVector(Vector3 vector) const
{
	return (*this * Quaternion(vector) * inversed()).toVector3();
}

__host__ __device__ Quaternion Quaternion::fromAngle(number_t angle, Vector3 axis)
{
	return Quaternion(cosf(angle / 2), 0, 0, 0) + Quaternion(axis) * sinf(angle / 2);
}
