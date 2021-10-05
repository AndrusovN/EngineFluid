#include "Transform.cuh"
#include "assert.cuh"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

__host__ __device__ Transform::Transform(GameObject* parent) : Component(parent)
{
	_position = Vector3::ZERO();
	_rotation = Quaternion::IDENTITY();
}

__host__ __device__ int Transform::typeId() const
{
	return TRANSFORM_TYPE_ID;
}

__host__ __device__ void Transform::rotate(Quaternion rotation)
{
	assert(rotation.magnitude() > EPSILON);
	rotation = rotation / rotation.magnitude();

	Vector<IComponent>* components = gameObject()->getComponents();
	for (int i = 0; i < components->size(); i++)
	{
		if ((*components)[i] == this) continue;
		Component* c = (Component*)(*components)[i];
		c->rotate(rotation);
	}

	_forward = rotation.applyToVector(_forward);
	_right = rotation.applyToVector(_right);
	_up = rotation.applyToVector(_up);

	_rotation *= rotation;
}

__host__ __device__ void Transform::translate(Vector3 offset)
{
	Vector<IComponent>* components = gameObject()->getComponents();
	for (int i = 0; i < components->size(); i++)
	{
		if ((*components)[i] == this) continue;
		Component* c = (Component*)(*components)[i];
		c->translate(offset);
	}

	_position += offset;
}

__host__ __device__ const Vector3 Transform::position() const
{
	return _position;
}

__host__ __device__ const Quaternion Transform::rotation() const
{
	return _rotation;
}

__host__ __device__ void Transform::setPosition(Vector3 position)
{
	translate(position - _position);
}

__host__ __device__ void Transform::setRotation(Quaternion rotation)
{
	rotate(_rotation.inversed() * rotation);
}

__host__ __device__ Vector3 Transform::forward()
{
	return _forward;
}

__host__ __device__ Vector3 Transform::back()
{
	return _forward * -1;
}

__host__ __device__ Vector3 Transform::right()
{
	return _right;
}

__host__ __device__ Vector3 Transform::left()
{
	return _right * -1;
}

__host__ __device__ Vector3 Transform::up()
{
	return _up;
}

__host__ __device__ Vector3 Transform::down()
{
	return _up * -1;
}

__host__ void Transform::moveToHost()
{
	_isOnDevice = false;
}

__host__ void Transform::moveToDevice()
{
	_isOnDevice = true;
}
