#include "GameObject.cuh"
#include <assert.h>
#include "Component.cuh"

void GameObject::addComponent(IComponent* component)
{
	assert(instanceof<Component>(component));
	
	((Component*)component)->assignToGameObject(this);
	_components.push_back(component);
}

void GameObject::removeComponent(IComponent* component)
{
	for (int i = 0; i < _components.size(); i++)
	{
		if (component == _components[i]) {
			_components.erase(_components.begin() + i);
			return;
		}
	}
}

std::vector<IComponent*> GameObject::getComponents()
{
	std::vector<IComponent*> result(_components.size());
	std::copy(_components.begin(), _components.end(), result);

	return result;
}

