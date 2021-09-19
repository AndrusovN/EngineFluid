#ifndef SCENE
#define SCENE

#include "GameObject.cuh"
#include "Component.cuh"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

typedef int TypeId;


class Scene : public IMovable {
private:
	thrust::host_vector<GameObject*> _gameObjectsHost;
	thrust::host_vector<Component*> _componentsHost;

	thrust::device_vector<GameObject*> _gameObjectsDevice;
	thrust::device_vector<Component*> _componentsDevice;

	bool _isOnDevice = false;

	void recalculateComponents();
public:
	Scene();
	Scene(thrust::host_vector<GameObject*> gameObjects);

	template <typename Type>
	vector<Type*>* getComponents() {
		vector<Type*> result;

		if (_isOnDevice) {
			result = new thrust::device_vector<Type*>();
		}
		else {
			result = new thrust::host_vector<Type*>();
		}

		Component* test = (Component*)(new Type());
		TypeId id = (test->typeId();
		delete test;

		vector<Component*>* components = _isOnDevice ? (vector<Component*>*)&_componentsDevice : (vector<Component*>*)&_componentsHost;

		for (int i = 0; i < components->size(); i++)
		{
			if ((*components)[i]->typeId() == id) {
				result->push_back((*components)[i]);
			}
		}

		return result;
	}

	template <typename Type>
	Type* getComponent() {
		Component* test = (Component*)(new Type());
		TypeId id = (test->typeId();
		delete test;

		vector<Component*>* components = _isOnDevice ? &_componentsDevice : &_componentsHost;

		for (int i = 0; i < components->size(); i++)
		{
			if ((*components)[i]->typeId() == id) {
				return (*components)[i]);
			}
		}
		
		return nullptr;
	}

	vector<GameObject*>* gameObjects();
	vector<Component*>* components();

	__host__ void moveToHost() override;
	__host__ void moveToDevice() override;
};

#endif
