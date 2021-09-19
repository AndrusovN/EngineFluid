#include "Transform.cuh"
#include "assert.cuh"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

Transform::Transform(GameObject* parent) : Component(parent)
{
	_position = Vector3::ZERO;
	_rotation = Quaternion::IDENTITY;
}

const int Transform::typeId() const
{
	return TRANSFORM_TYPE_ID;
}

void Transform::rotate(Quaternion rotation)
{
	assert(rotation.magnitude() > EPSILON);
	rotation = rotation / rotation.magnitude();

	if (_isOnDevice) {
		thrust::device_vector<Component*> components = gameObject()->getComponentsDevice();
		for (int i = 0; i < components.size(); i++)
		{
			if (components[i] == this) continue;
			Component* c = components[i];
			c->rotate(rotation);
		}
	}
	else {
		thrust::host_vector<Component*> components = gameObject()->getComponentsHost();
		for (int i = 0; i < components.size(); i++)
		{
			if (components[i] == this) continue;
			components[i]->rotate(rotation);
		}
	}

	_forward = rotation.applyToVector(_forward);
	_right = rotation.applyToVector(_right);
	_up = rotation.applyToVector(_up);

	_rotation *= rotation;
}

void Transform::translate(Vector3 offset)
{
	if (_isOnDevice) {
		thrust::device_vector<Component*> components = gameObject()->getComponentsDevice();
		for (int i = 0; i < components.size(); i++)
		{
			if (components[i] == this) continue;
			Component* c = components[i];
			c->translate(offset);
		}
	}
	else {
		thrust::host_vector<Component*> components = gameObject()->getComponentsHost();
		for (int i = 0; i < components.size(); i++)
		{
			if (components[i] == this) continue;
			components[i]->translate(offset);
		}
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

void Transform::moveToHost()
{
	_isOnDevice = false;
}

void Transform::moveToDevice()
{
	_isOnDevice = true;
}
