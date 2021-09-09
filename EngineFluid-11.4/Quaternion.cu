#include "Quaternion.cuh"
#include "assert.h"

const Quaternion Quaternion::IDENTITY = Quaternion(1, 0, 0, 0);

Quaternion::Quaternion(number_t x, number_t y, number_t z, number_t w) : _x(x), _y(y), _z(z), _w(w) {}

Quaternion::Quaternion() : _x(0), _y(0), _z(0), _w(0) {}

Quaternion::Quaternion(const Vector3& base) : _x(0), _y(base.x()), _z(base.y()), _w(base.z()) {}

const Quaternion Quaternion::operator*(number_t value) const
{
	return Quaternion(_x * value, _y * value, _z * value, _w * value);
}

const Quaternion Quaternion::operator/(number_t value) const
{
	assert(value > EPSILON);
	return *this * ((number_t)1 / value);
}

const Quaternion Quaternion::operator+(const Quaternion& other) const
{
	return Quaternion(_x + other._x, _y + other._y, _z + other._z, _w + other._w);
}

const Quaternion Quaternion::operator-(const Quaternion& other) const
{
	return *this + (other * -1);
}

const Quaternion Quaternion::operator*(const Quaternion& other) const
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

const bool Quaternion::operator==(const Quaternion& other) const
{
	return _x == other._x &&
		_y == other._y &&
		_z == other._z &&
		_w == other._w;
}

const bool Quaternion::operator!=(const Quaternion& other) const
{
	return !(*this == other);
}

const Quaternion Quaternion::conjugated() const
{
	return Quaternion(_x, -_y, -_z, -_w);
}

const Quaternion Quaternion::inversed() const
{
	return conjugated() / sqrMagnitude();
}

const number_t Quaternion::magnitude() const
{
	return SQRT(sqrMagnitude());
}

const number_t Quaternion::sqrMagnitude() const
{
	return (*this * conjugated())._x;
}

const Vector3 Quaternion::toVector3() const
{
	return Vector3(_y, _z, _w);
}

const Vector3 Quaternion::applyToVector(Vector3 vector) const
{
	return (*this * Quaternion(vector) * inversed()).toVector3();
}

Quaternion Quaternion::fromAngle(number_t angle, Vector3 axis)
{
	return Quaternion(cosf(angle / 2), 0, 0, 0) + Quaternion(axis) * sinf(angle / 2);
}
