#include "Scene.cuh"
#include "assert.h"


__host__ __device__ void Scene::recalculateComponents()
{
	for (int i = 0; i < _gameObjects.size(); i++)
	{
		Vector<IComponent>* goComponents = _gameObjects[i]->getComponents();

		for (int j = 0; j < goComponents->size(); j++)
		{
			_components.push((Component*)(IComponent*)(*goComponents)[j]);
		}
	}
}

__host__ __device__ Scene::Scene() {}

__host__ __device__ Scene::Scene(Vector<GameObject> gameObjects)
{
	_gameObjects = *(Vector<GameObject>*)(gameObjects.copy());

	recalculateComponents();
}

__host__ __device__ Vector<GameObject>* Scene::gameObjects()
{
	return (Vector<GameObject>*)_gameObjects.copy();
}

__host__ __device__ Vector<Component>* Scene::components()
{
	return (Vector<Component>*)_components.copy();
}

__host__ void Scene::moveToHost()
{
	_isOnDevice = false;
	_gameObjects.moveToHost();

	recalculateComponents();
}

__host__ void Scene::moveToDevice()
{
	_isOnDevice = true;
	_gameObjects.moveToDevice();

	recalculateComponents();
}
