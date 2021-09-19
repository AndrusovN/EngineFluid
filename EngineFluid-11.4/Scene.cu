#include "Scene.cuh"
#include "assert.h"


void Scene::recalculateComponents()
{
	vector<Component*>* components = (_isOnDevice ? (vector<Component*>*)(&_componentsDevice) : (vector<Component*>*)(&_componentsHost));

	vector<GameObject*>* gameObjects = _isOnDevice ? (vector<GameObject*>*) & _gameObjectsDevice : (vector<GameObject*>*) & _gameObjectsHost;

	for (int i = 0; i < gameObjects->size(); i++)
	{
		if (_isOnDevice) {
			thrust::device_vector<Component*> goComponents = (*gameObjects)[i]->getComponentsDevice();

			for (int j = 0; j < goComponents.size(); j++)
			{
				components->push_back(goComponents[j]);
			}
		}
		else {
			thrust::host_vector<Component*> goComponents = (*gameObjects)[i]->getComponentsHost();

			for (int j = 0; j < goComponents.size(); j++)
			{
				components->push_back(goComponents[j]);
			}
		}
	}
}

Scene::Scene() {}

Scene::Scene(thrust::host_vector<GameObject*> gameObjects)
{
	_gameObjectsHost = thrust::host_vector<GameObject*>(gameObjects.size());
	thrust::copy(gameObjects.begin(), gameObjects.end(), _gameObjectsHost.begin());

	recalculateComponents();
}

vector<GameObject*>* Scene::gameObjects()
{
	vector<GameObject*>* result;
	if (_isOnDevice) {
		result = (vector<GameObject*>*)(new thrust::device_vector<GameObject*>());
		thrust::copy(_gameObjectsDevice.begin(), _gameObjectsDevice.end(), result->begin());
	}
	else {
		result = (vector<GameObject*>*)(new thrust::host_vector<GameObject*>());
		thrust::copy(_gameObjectsHost.begin(), _gameObjectsHost.end(), result->begin());
	}
	
	return result;
}

vector<Component*>* Scene::components()
{
	vector<Component*>* result;
	if (_isOnDevice) {
		result = (vector<Component*>*)(new thrust::device_vector<Component*>());
		thrust::copy(_componentsDevice.begin(), _componentsDevice.end(), result->begin());
	}
	else {
		result = (vector<Component*>*)(new thrust::host_vector<Component*>());
		thrust::copy(_componentsHost.begin(), _componentsHost.end(), result->begin());
	}

	return result;
}

void Scene::moveToHost()
{
	_isOnDevice = false;
	_gameObjectsHost = thrust::host_vector<GameObject*>(_gameObjectsDevice.size());
	for (int i = 0; i < _gameObjectsDevice.size(); i++)
	{
		GameObject* go = new GameObject();
		cudaMemcpy(go, _gameObjectsDevice[i], sizeof(GameObject), cudaMemcpyDeviceToHost);
		cudaFree(_gameObjectsDevice[i]);
		go->moveToHost();
		
		_gameObjectsHost[i] = go;
	}

	recalculateComponents();
}

void Scene::moveToDevice()
{
	_isOnDevice = true;
	_gameObjectsDevice = thrust::device_vector<GameObject*>(_gameObjectsHost.size());
	for (int i = 0; i < _gameObjectsHost.size(); i++)
	{
		GameObject* go = nullptr;
		cudaMalloc(&go, sizeof(GameObject));
		cudaMemcpy(go, _gameObjectsHost[i], sizeof(GameObject), cudaMemcpyHostToDevice);
		delete _gameObjectsHost[i];
		go->moveToDevice();

		_gameObjectsDevice[i] = go;
	}

	recalculateComponents();
}
