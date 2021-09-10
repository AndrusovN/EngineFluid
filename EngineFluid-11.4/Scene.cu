#include "Scene.cuh"
#include <utility>
#include <assert.h>


Scene::Scene() {}

Scene::Scene(std::vector<GameObject*> gameObjects)
{
	_gameObjects = std::vector<GameObject*>(gameObjects.size());
	std::copy(gameObjects.begin(), gameObjects.end(), _gameObjects);

	_components = std::multimap<TypeId, Component*>();
	for (auto gameObject : gameObjects)
	{
		for (auto component : gameObject->getComponents())
		{
			Component* _component = static_cast<Component*>(component);
			assert(_component != nullptr);

			_components.insert(std::pair<TypeId, Component*>(component->typeId(), _component));
		}
	}
}

std::vector<GameObject*> Scene::gameObjects()
{
	std::vector<GameObject*> result(_gameObjects.size());
	std::copy(_gameObjects.begin(), _gameObjects.end(), result);

	return result;
}

std::vector<Component*> Scene::components()
{
	std::vector<Component*> result(_components.size());
	std::copy(_components.begin(), _components.end(), result);

	return result;
}
