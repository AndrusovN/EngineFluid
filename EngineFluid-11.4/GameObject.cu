#include "GameObject.cuh"
#include "assert.cuh"
#include "Component.cuh"

__host__ __device__ void GameObject::addComponent(IComponent* component)
{
	((Component*)component)->resetGameObject(this);

	_components.push(component);

}

__host__ __device__ void GameObject::removeComponent(IComponent* component)
{
	for (int i = 0; i < _components.size(); i++)
	{
		if (component == _components[i]) {
			_components.remove(i);
			return;
		}
	}
}

__host__ __device__ Vector<IComponent>* GameObject::getComponents()
{
	return (Vector<IComponent>*)(_components.copy());
}

__host__ void GameObject::moveToHost()
{
	_isOnDevice = false;
	_components.moveToHost();

	for (int i = 0; i < _components.size(); i++)
	{
		((Component*)_components[i])->resetGameObject(this);
	}
}

__host__ void GameObject::moveToDevice()
{
	_isOnDevice = true;
	_components.moveToDevice();

	for (int i = 0; i < _components.size(); i++)
	{
		((Component*)_components[i])->resetGameObject(this);
	}
}
