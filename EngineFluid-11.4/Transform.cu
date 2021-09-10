#include "Transform.cuh"
#include <assert.h>
#include <typeinfo>

int Transform::typeId()
{
	return typeid(Transform).hash_code();
}

void Transform::rotate(Quaternion rotation)
{
	assert(rotation.magnitude() > EPSILON);
	rotation = rotation / rotation.magnitude();

	for (auto _component : gameObject()->getComponents())
	{
		if ((void*)_component == (void*)this) {
			continue;
		}
		_component->rotate(rotation);
	}

	_forward = rotation.applyToVector(_forward);
	_right = rotation.applyToVector(_right);
	_up = rotation.applyToVector(_up);

	_rotation *= rotation;
}

void Transform::translate(Vector3 offset)
{
	for (auto _component : gameObject()->getComponents())
	{
		if ((void*)_component == (void*)this) {
			continue;
		}
		_component->translate(offset);
	}

	_position += offset;
}

const Vector3 Transform::position() const
{
	return _position;
}

const Quaternion Transform::rotation() const
{
	return _rotation;
}

void Transform::setPosition(Vector3 position)
{
	translate(position - _position);
}

void Transform::setRotation(Quaternion rotation)
{
	rotate(_rotation.inversed() * rotation);
}

Vector3 Transform::forward()
{
	return _forward;
}

Vector3 Transform::back()
{
	return _forward * -1;
}

Vector3 Transform::right()
{
	return _right;
}

Vector3 Transform::left()
{
	return _right * -1;
}

Vector3 Transform::up()
{
	return _up;
}

Vector3 Transform::down()
{
	return _up * -1;
}
