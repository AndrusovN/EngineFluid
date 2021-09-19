#ifndef TRANSFORM
#define TRANSFORM

#include "Component.cuh"

const int TRANSFORM_TYPE_ID = 4;

class Transform : public Component {
private:
	Vector3 _position;
	Quaternion _rotation;

	Vector3 _forward = Vector3(0, 0, 1),
		_right = Vector3(1, 0, 0),
		_up = Vector3(0, 1, 0);

	bool _isOnDevice = false;;
public:
	Transform(GameObject* parent);

	__host__ __device__ const int typeId() const override;

	__host__ __device__ void rotate(Quaternion rotation) override;
	__host__ __device__ void translate(Vector3 offset) override;

	__host__ __device__ const Vector3 position() const;
	__host__ __device__ const Quaternion rotation() const;

	__host__ __device__ void setPosition(Vector3 position);
	__host__ __device__ void setRotation(Quaternion rotation);

	__host__ __device__ Vector3 forward();
	__host__ __device__ Vector3 back();
	__host__ __device__ Vector3 right();
	__host__ __device__ Vector3 left();
	__host__ __device__ Vector3 up();
	__host__ __device__ Vector3 down();

	__host__ void moveToHost() override;
	__host__ void moveToDevice() override;
};

#endif
