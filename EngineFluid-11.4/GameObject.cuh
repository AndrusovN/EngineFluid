#ifndef GAME_OBJECT
#define GAME_OBJECT

#include "IComponent.cuh"
#include "Vector.cuh"

#define PASS #ERROR("Define this part of code")

template <typename Base, typename T>
__host__ __device__ inline bool instanceof(const T* object) {
	IComponent* basetest = (IComponent*)(new Base());
	bool result = basetest->typeId() == ((IComponent*)object)->typeId();
	delete basetest;

	return result;
}

class GameObject : public IMovable {
private:
	Vector<IComponent> _components;

	bool _isOnDevice = false;
public:
	template <typename Type>
	__host__ __device__ Type* getComponentOfType() {
		for (int i = 0; i < _components.size(); i++)
		{
			IComponent* component = _components[i];
			if (instanceof<Type>(component)) {
				return (Type*)component;
			}
		}
		
		return nullptr;
	}

	template <typename Type>
	__host__ __device__ Vector<Type> getComponentsOfType() {
		Vector<Type> result;
		for (int i = 0; i < _components.size(); i++)
		{
			auto component = _components[i];
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

	__host__ __device__ Vector<IComponent>* getComponents();

	__host__ void moveToHost() override;
	__host__ void moveToDevice() override;
};

#endif
