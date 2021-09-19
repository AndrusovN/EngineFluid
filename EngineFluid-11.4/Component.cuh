#ifndef COMPONENT
#define COMPONENT

#include "GameObject.cuh"

class Component : public IComponent {
private:
	GameObject* _gameObject;
public:
	Component() : _gameObject(nullptr) {}

	Component(GameObject* parent) : _gameObject(parent) {}

	GameObject* gameObject() {
		return _gameObject;
	}

	__host__ __device__ virtual void resetGameObject(GameObject* object) {
		_gameObject = object;
	}

	__host__ __device__ void start() override {}
	__host__ __device__ void update() override {}
	__host__ __device__ void awake() override {}
	__host__ __device__ void deviceUpdate() override {}
	__host__ __device__ void rotate(Quaternion rotation) override {}
	__host__ __device__ void translate(Vector3 offset) override {}
	__host__ __device__ void onDestroy() override {}
};

#endif