#ifndef GAME_MANAGER
#define GAME_MANAGER

#include "Scene.cuh"
#include "Drawing.h"
#include "VisualEffect.cuh"
#include <mutex>

class GameManager {
private:
	static GameManager* _instance;
	std::vector<Scene*> _scenes;
	std::vector<VisualEffect*> _visualEffects;

	std::mutex _fieldsMutex;

	bool _stopped = false;
	int _currentSceneId;
	UIDrawer* drawer;
public:
	GameManager(UIDrawer* drawer, std::vector<Scene*> scenes);

	void addVisualEffect(VisualEffect* effect);

	void removeVisualEffect(VisualEffect* effect);

	static GameManager* instance();

	Scene* getScene(int index);

	Scene* getCurrentScene();

	void stop();

	void run();

	void changeScene(int sceneIndex);
};

#endif
