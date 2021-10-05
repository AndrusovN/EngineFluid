#ifndef COMPONENT
#define COMPONENT

#include "GameObject.cuh"

class Component : public IComponent {
private:
	GameObject* _gameObject;
public:
	__host__ __device__ Component() : _gameObject(nullptr) {}

	__host__ __device__ Component(GameObject* parent) : _gameObject(parent) {}

	__host__ __device__ GameObject* gameObject() {
		return _gameObject;
	}

	__host__ __device__ virtual void resetGameObject(GameObject* object) {
		_gameObject = object;
	}

	__host__ __device__ void start() override {}
	__host__ __device__ void update() override {}
	__host__ __device__ void awake() override {}
	__host__ void deviceUpdate() override {}
	__host__ __device__ void rotate(Quaternion rotation) override {}
	__host__ __device__ void translate(Vector3 offset) override {}
	__host__ __device__ void onDestroy() override {}
};

#endif