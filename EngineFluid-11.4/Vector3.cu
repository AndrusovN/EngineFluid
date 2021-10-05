#include "cuda_runtime.h"
#include "Vector3.cuh"
#include "assert.cuh"


__host__ __device__ Vector3::Vector3() : _x(0), _y(0), _z(0) {}

__host__ __device__ Vector3::Vector3(number_t x, number_t y, number_t z) : _x(x), _y(y), _z(z) {}

__host__ __device__ Vector3::~Vector3() {}

__host__ __device__ Vector3 Vector3::operator = (const Vector3& other) {
	_x = other._x;
	_y = other._y;
	_z = other._z;

	return *this;
}

__host__ __device__ number_t Vector3::x() const {
	return _x;
}

__host__ __device__ number_t Vector3::y() const {
	return _y;
}

__host__ __device__ number_t Vector3::z() const {
	return _z;
}

__host__ __device__ void Vector3::set_x(number_t x) {
	_x = x;
}

__host__ __device__ void Vector3::set_y(number_t y) {
	_y = y;
}

__host__ __device__ void Vector3::set_z(number_t z) {
	_z = z;
}

__host__ __device__ void Vector3::normalize() {
	assert(magnitude() > EPSILON);
	*this /= magnitude();
}

__host__ __device__ number_t Vector3::magnitude() const
{
	return SQRT(sqrMagnitude());
}

__host__ __device__ number_t Vector3::sqrMagnitude() const
{
	return this->dot(*this);
}

__host__ __device__ number_t* Vector3::toCArray() const
{
	number_t* result = new number_t[3];
	result[0] = _x;
	result[1] = _y;
	result[2] = _z;
	return result;
}

__host__ __device__ bool Vector3::operator==(const Vector3& other) const
{
	return	_x == other._x &&
			_y == other._y &&
			_z == other._z;
}

__host__ __device__ bool Vector3::operator!=(const Vector3& other) const
{
	return !(*this == other);
}

__host__ __device__ const Vector3 Vector3::operator*(const number_t value) const
{
	return Vector3(_x * value, _y * value, _z * value);
}

__host__ __device__ const Vector3 Vector3::operator/(const number_t value) const
{
	assert(value < -EPSILON || EPSILON < value);
	return (*this * ((number_t)1.0 / value));
}

__host__ __device__ const Vector3 Vector3::operator+(const Vector3& other) const
{
	return Vector3(_x + other._x, _y + other._y, _z + other._z);
}

__host__ __device__ const Vector3 Vector3::operator-(const Vector3& other) const
{
	return *this + (other * -1);
}

__host__ __device__ Vector3 Vector3::operator*=(number_t value)
{
	return *this = *this * value;
}

__host__ __device__ Vector3 Vector3::operator/=(number_t value)
{
	return *this = *this * value;
}

__host__ __device__ Vector3 Vector3::operator+=(const Vector3& other)
{
	return *this = *this + other;
}

__host__ __device__ Vector3 Vector3::operator-=(const Vector3& other)
{
	return *this = *this - other;
}

__host__ __device__ const Vector3 Vector3::normalized()
{
	return *this / magnitude();
}

__host__ __device__ number_t Vector3::dot(const Vector3& other) const
{
	return _x * other._x + _y * other._y + _z * other._z;
}

__host__ __device__ const Vector3 Vector3::cross(const Vector3& other) const
{
	return Vector3(_y * other._z - _z * other._y,
		_z * other._x - _x * other._z,
		_x * other._y - _y * other._x);
}

__host__ __device__ const Vector3 Vector3::multiplyComponentWise(const Vector3& other) const
{
	return Vector3(_x * other._x, _y * other._y, _z * other._z);
}

__host__ __device__ number_t Vector3::angle(const Vector3& other) const
{
	return ATAN2(angle_sin(other), angle_cos(other));
}

__host__ __device__ number_t Vector3::unsignedAngle(const Vector3& other) const
{
	return abs(angle(other));
}

__host__ __device__ number_t Vector3::angle_sin(const Vector3& other) const
{
	return cross(other).magnitude() / magnitude() / other.magnitude();
}

__host__ __device__ number_t Vector3::angle_cos(const Vector3& other) const
{
	return dot(other) / magnitude() / other.magnitude();
}

__host__ __device__ const Vector3 Vector3::reflect(const Vector3& planeNormal) const
{
	return planeNormal * (2 * angle_cos(planeNormal)) * magnitude() + *this;
}

__host__ __device__ number_t Vector3::distance(const Vector3& a, const Vector3& b)
{
	return (a - b).magnitude();
}

__host__ __device__ number_t Vector3::sqrDistance(const Vector3& a, const Vector3& b)
{
	return (a - b).sqrMagnitude();
}
