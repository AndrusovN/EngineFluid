#include "GameObject.cuh"
#include "assert.cuh"
#include "Component.cuh"

void GameObject::addComponent(IComponent* component)
{
	assert(instanceof<Component>(component));
	
	((Component*)component)->resetGameObject(this);

	if (_isOnDevice) {
		_componentsDevice.push_back(component);
	}
	else {
		_componentsHost.push_back(component);
	}
	
}

void GameObject::removeComponent(IComponent* component)
{
	for (int i = 0; i < _isOnDevice ? _componentsDevice.size() : _componentsHost.size(); i++)
	{
		if (_isOnDevice) {
			if (component == _componentsDevice[i]) {
				_componentsDevice.erase(_componentsDevice.begin() + i);
				return;
			}
		}
		else {
			if (component == _componentsHost[i]) {
				_componentsHost.erase(_componentsHost.begin() + i);
				return;
			}
		}
		
	}
}

thrust::device_vector<IComponent*> GameObject::getComponentsDevice()
{
	thrust::device_vector<IComponent*> result(_componentsDevice.size());
	thrust::copy(_componentsDevice.begin(), _componentsDevice.end(), result.begin());

	return result;
}

thrust::host_vector<IComponent*> GameObject::getComponentsHost()
{
	thrust::host_vector<IComponent*> result(_componentsHost.size());
	thrust::copy(_componentsHost.begin(), _componentsHost.end(), result.begin());

	return result;
}

void GameObject::moveToHost()
{
	_isOnDevice = false;
	_componentsHost = thrust::host_vector<IComponent*>(_componentsDevice.size());
	for (int i = 0; i < _componentsDevice.size(); i++)
	{
		int size = sizeof(*_componentsDevice[i]);

		Component* component = (Component*)malloc(size);
		cudaMemcpy(component, _componentsDevice[i], size, cudaMemcpyDeviceToHost);
		cudaFree(_componentsDevice[i]);

		component->moveToHost();
		component->resetGameObject(this);
		_componentsHost[i] = component;
	}
}

void GameObject::moveToDevice()
{
	_isOnDevice = true;
	_componentsDevice = thrust::device_vector<IComponent*>(_componentsHost.size());
	for (int i = 0; i < _componentsHost.size(); i++)
	{
		int size = sizeof(*_componentsHost[i]);

		Component* component = nullptr;
		cudaMalloc(&component, size);
		cudaMemcpy(component, _componentsHost[i], size, cudaMemcpyHostToDevice);
		delete _componentsHost[i];

		component->moveToDevice();
		component->resetGameObject(this);
		_componentsDevice[i] = component;
	}
}

