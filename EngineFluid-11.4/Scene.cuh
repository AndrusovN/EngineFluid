#ifndef SCENE
#define SCENE

#include "GameObject.cuh"
#include "Component.cuh"
#include <map>
#include <typeinfo>

typedef int TypeId;

class Scene {
private:
	std::vector<GameObject*> _gameObjects;
	std::multimap<TypeId, Component*> _components;

public:
	Scene();
	Scene(std::vector<GameObject*> gameObjects);

	template <typename Type>
	std::vector<Type*> getComponents() {
		std::vector<Type*> result;

		TypeId id = typeid(Type).hash_code();
		for (auto ptr = _components.lower_bound(id); 
			ptr != _components.end() && instanceof<Type>(*ptr); 
			ptr++)
		{
			result.push_back((Type*)*ptr);
		}

		return result;
	}

	template <typename Type>
	Type* getComponent() {
		TypeId id = typeid(Type);
		auto ptr = _components.lower_bound(id);
		if (ptr != _components.end() && instanceof<Type>(*ptr)) {
			return *ptr;
		}
		return nullptr;
	}

	std::vector<GameObject*> gameObjects();
	std::vector<Component*> components();
};

#endif
