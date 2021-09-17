#include "GameManager.cuh"
#include <assert.h>

GameManager* GameManager::_instance = nullptr;

GameManager::GameManager(UIDrawer* drawer, std::vector<Scene*> scenes)
{
	assert(_instance == nullptr);

	_instance = this;
	this->drawer = drawer;
	_scenes = scenes;
	_currentSceneId = 0;
}

void GameManager::addVisualEffect(VisualEffect* effect)
{
	assert(effect != nullptr);

	_visualEffects.push_back(effect);
}

void GameManager::removeVisualEffect(VisualEffect* effect)
{
	for (int i = 0; i < _visualEffects.size(); i++)
	{
		if ((void*)_visualEffects[i] == (void*)effect) {
			_visualEffects.erase(_visualEffects.begin() + i);
			return;
		}
	}
}

GameManager* GameManager::instance()
{
	return _instance;
}

Scene* GameManager::getScene(int index)
{
	assert(0 <= index && index < _scenes.size());

	return _scenes[index];
}

Scene* GameManager::getCurrentScene()
{
	return _scenes[_currentSceneId];
}

void GameManager::stop()
{
	_stopped = true;
}

void GameManager::run()
{
	assert(scenes.size() > 0);

	changeScene(0);

	while (!_stopped) {
		_fieldsMutex.lock();
		for (auto component : _scenes[_currentSceneId]->components())
		{
			component->update();
		}
		for (auto component : _scenes[_currentSceneId]->components())
		{
			component->lateUpdate();
		}
		_fieldsMutex.unlock();

		Color* colorMap = drawer->getColorMap();
		int width = drawer->getWidth();
		int height = drawer->getHeight();

		for (auto effect : _visualEffects)
		{
			effect->apply(colorMap, width, height, _scenes[_currentSceneId]);
		}

		drawer->display();
	}
}

void GameManager::changeScene(int sceneIndex)
{
	assert(0 <= sceneIndex && sceneIndex < _scenes.size());

	_fieldsMutex.lock();

	_currentSceneId = sceneIndex;

	for (auto component : _scenes[_currentSceneId]->components())
	{
		component->awake();
	}

	for (auto component : _scenes[_currentSceneId]->components())
	{
		component->start();
	}

	_fieldsMutex.unlock();
}
