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

	void assignToGameObject(GameObject* gameObject) {
		_gameObject = gameObject;
	}

	void start() override {}
	void update() override {}
	void awake() override {}
	void lateUpdate() override {}
	void rotate(Quaternion rotation) override {}
	void translate(Vector3 offset) override {}
	void onDestroy() override {}
};

#endif