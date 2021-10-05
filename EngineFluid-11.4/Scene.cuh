#ifndef SCENE
#define SCENE

#include "GameObject.cuh"
#include "Component.cuh"
#include "Vector.cuh"

typedef int TypeId;


class Scene : public IMovable {
private:
	Vector<GameObject> _gameObjects;
	Vector<Component> _components;

	bool _isOnDevice = false;

	__host__ __device__ void recalculateComponents();
public:
	__host__ __device__ Scene();
	__host__ __device__ Scene(Vector<GameObject> gameObjects);

	template <typename Type>
	__host__ __device__ Vector<Type>* getComponents() {
		Vector<Type>* result = new Vector<Type>();

		Component* test = (Component*)(new Type());
		TypeId id = (test->typeId());
		delete test;

		for (int i = 0; i < _components.size(); i++)
		{
			if (_components[i]->typeId() == id) {
				result->push((Type*)(_components[i]));
			}
		}

		return result;
	}

	template <typename Type>
	__host__ __device__ Type* getComponent() {
		Component* test = (Component*)(new Type());
		TypeId id = (test->typeId());
		delete test;

		for (int i = 0; i < _components.size(); i++)
		{
			if (_components[i]->typeId() == id) {
				return _components[i];
			}
		}
		
		return nullptr;
	}

	__host__ __device__ Vector<GameObject>* gameObjects();
	__host__ __device__ Vector<Component>* components();

	__host__ void moveToHost() override;
	__host__ void moveToDevice() override;
};

#endif
