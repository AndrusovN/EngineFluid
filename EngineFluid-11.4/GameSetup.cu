#include "GameSetup.cuh"
#include "GameManager.cuh"
#include "Camera.cuh"
#include "Mesh.cuh"
#include "Light.cuh"
#include "Triangle.cuh"
#include "GameObject.cuh"

void startGame(UIDrawer* drawer)
{
	GameObject camObj = GameObject();
	Transform camTransform = Transform(&camObj);

	Camera cam = Camera(&camObj, drawer, { 0, 0, drawer->getWidth(), drawer->getHeight() }, 60.f, 40.0f, 0.1f, 100.0f);
	
	camObj.addComponent(&cam);
	camObj.addComponent(&camTransform);
	camTransform.setPosition(Vector3(0, 0, -5));

	GameObject cubeObj = GameObject();
	Transform cubeTransform = Transform(&cubeObj);
	Mesh cubeMesh = Mesh(&cubeObj, "Assets\\Cube.obj", 15);
	cubeMesh.moveToCUDA();

	cubeObj.addComponent(&cubeMesh);
	cubeObj.addComponent(&cubeTransform);

	GameObject lightObj = GameObject();
	Transform lightTransform = Transform(&lightObj);
	GeneralLight light = GeneralLight(&lightObj);

	lightTransform.setPosition(Vector3(-2, 5, -1));
	lightTransform.rotate(Quaternion::fromAngle(PI / 2, Vector3::RIGHT));

	std::vector<GameObject*> gameObjects = { &camObj, &cubeObj, &lightObj };

	Scene mainScene = Scene(gameObjects);

	GameManager* manager = new GameManager(drawer, { &mainScene });

	manager->run();
}

