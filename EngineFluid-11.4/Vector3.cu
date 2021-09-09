#include "cuda_runtime.h"
#include "Vector3.cuh"
#include <assert.h>

const	Vector3 Vector3::UP = Vector3(0, 1, 0), 
		Vector3::DOWN = Vector3(0, -1, 0), 
		Vector3::LEFT = Vector3(-1, 0, 0), 
		Vector3::RIGHT = Vector3(1, 0, 0), 
		Vector3::FORWARD = Vector3(0, 0, 1), 
		Vector3::BACK = Vector3(0, 0, -1), 
		Vector3::ZERO = Vector3(0, 0, 0),
		Vector3::INFINITY = Vector3((number_t)NUMBER_T_INFTY, (number_t)NUMBER_T_INFTY, (number_t)NUMBER_T_INFTY);

Vector3::Vector3() : _x(0), _y(0), _z(0) {}

Vector3::Vector3(number_t x, number_t y, number_t z) : _x(x), _y(y), _z(z) {}

Vector3::~Vector3() {}

Vector3 Vector3::operator = (const Vector3& other) {
	_x = other._x;
	_y = other._y;
	_z = other._z;
}

const number_t Vector3::x() {
	return _x;
}

const number_t Vector3::y() {
	return _y;
}

const number_t Vector3::z() {
	return _z;
}

void Vector3::set_x(number_t x) {
	_x = x;
}

void Vector3::set_y(number_t y) {
	_y = y;
}

void Vector3::set_z(number_t z) {
	_z = z;
}

void Vector3::set_epsilon(number_t epsilon)
{
	assert(epsilon >= 0);
	EPSILON = epsilon;
}

void Vector3::normalize() {
	assert(magnitude() > EPSILON);
	*this /= magnitude();
}

const number_t Vector3::magnitude() const
{
	return SQRT(sqrMagnitude());
}

const number_t Vector3::sqrMagnitude() const
{
	return this->dot(*this);
}

const number_t* Vector3::toCArray() const
{
	number_t* result = new number_t[3];
	result[0] = _x;
	result[1] = _y;
	result[2] = _z;
	return result;
}

const bool Vector3::operator==(const Vector3& other) const
{
	return	_x == other._x &&
			_y == other._y &&
			_z == other._z;
}

const bool Vector3::operator!=(const Vector3& other) const
{
	return !(*this == other);
}

const Vector3 Vector3::operator*(const number_t value) const
{
	return Vector3(_x * value, _y * value, _z * value);
}

const Vector3 Vector3::operator/(const number_t value) const
{
	assert(value < -EPSILON || EPSILON < value);
	return (*this * ((number_t)1.0 / value));
}

const Vector3 Vector3::operator+(const Vector3& other) const
{
	return Vector3(_x + other._x, _y + other._y, _z + other._z);
}

const Vector3 Vector3::operator-(const Vector3& other) const
{
	return *this + (other * -1);
}

Vector3 Vector3::operator*=(number_t value)
{
	return *this = *this * value;
}

Vector3 Vector3::operator/=(number_t value)
{
	return *this = *this * value;
}

Vector3 Vector3::operator+=(const Vector3& other)
{
	return *this = *this + other;
}

Vector3 Vector3::operator-=(const Vector3& other)
{
	return *this = *this - other;
}

const Vector3 Vector3::normalized()
{
	return *this / magnitude();
}

const number_t Vector3::dot(const Vector3& other) const
{
	return _x * other._x + _y * other._y + _z * other._z;
}

const Vector3 Vector3::cross(const Vector3& other) const
{
	return Vector3(_y * other._z - _z * other._y,
		_z * other._x - _x * other._z,
		_x * other._y - _y * other._x);
}

const Vector3 Vector3::multiplyComponentWise(const Vector3& other) const
{
	return Vector3(_x * other._x, _y * other._y, _z * other._z);
}

const number_t Vector3::angle(const Vector3& other) const
{
	return ATAN2(angle_sin(other), angle_cos(other));
}

const number_t Vector3::unsignedAngle(const Vector3& other) const
{
	return abs(angle(other));
}

const number_t Vector3::angle_sin(const Vector3& other) const
{
	return cross(other).magnitude() / magnitude() / other.magnitude();
}

const number_t Vector3::angle_cos(const Vector3& other) const
{
	return dot(other) / magnitude() / other.magnitude();
}

const Vector3 Vector3::reflect(const Vector3& planeNormal) const
{
	return planeNormal * (2 * angle_cos(planeNormal)) * magnitude() + *this;
}

number_t Vector3::distance(const Vector3& a, const Vector3& b)
{
	return (a - b).magnitude();
}

__host__ __device__ number_t Vector3::sqrDistance(const Vector3& a, const Vector3& b)
{
	return (a - b).sqrMagnitude();
}
