#include "Quaternion.cuh"


Quaternion::Quaternion(number_t x, number_t y, number_t z, number_t w)
{
}

Quaternion::Quaternion()
{
}

Quaternion::Quaternion(const Vector3& base)
{
}

const Quaternion Quaternion::operator*(number_t value) const
{
	return Quaternion();
}

const Quaternion Quaternion::operator/(number_t value) const
{
	return Quaternion();
}

const Quaternion Quaternion::operator*(const Quaternion& other) const
{
	return Quaternion();
}

const bool Quaternion::operator==(const Quaternion& other) const
{
	return false;
}

const bool Quaternion::operator!=(const Quaternion& other) const
{
	return false;
}

const number_t Quaternion::magnitude() const
{
	return 0;
}

const number_t Quaternion::sqrMagnitude() const
{
	return 0;
}

const Vector3 Quaternion::toVector3() const
{
	return Vector3();
}

const Vector3 Quaternion::applyToVector(Vector3 vector) const
{
	return Vector3();
}
