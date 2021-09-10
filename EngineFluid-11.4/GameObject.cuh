#ifndef GAME_OBJECT
#define GAME_OBJECT

#include "IComponent.cuh"
#include <vector>

#define PASS ERROR("Define this part of code")

template <typename Base, typename T>
__host__ __device__ inline bool instanceof(const T* object) {
	return dynamic_cast<const Base*>(object) != nullptr;
}

class GameObject {
private:
	std::vector<IComponent*> _components;
public:
	template <typename Type>
	__host__ __device__ Type* getComponentOfType() {
		for (auto component : _components)
		{
			if (instanceof<Type>(component)) {
				return component;
			}
		}

		return nullptr;
	}

	template <typename Type>
	__host__ __device__ std::vector<Type*> getComponentsOfType() {
		std::vector<Type*> result;
		for (auto component : _components)
		{
			if (instanceof<Type>(component)) {
				result.push_back(component);
			}
		}

		return result;
	}

	__host__ __device__ bool isNull() {
		return this == nullptr;
	}

	__host__ __device__ void addComponent(IComponent* component);

	__host__ __device__ void removeComponent(IComponent* component);

	__host__ __device__ std::vector<IComponent*> getComponents();
};

#endif
