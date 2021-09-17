#ifndef CAMERA
#define CAMERA

#include "Component.cuh"
#include "Light.cuh"
#include "Scene.cuh"
#include "Triangle.cuh"
#include "Transform.cuh"
#include "Drawing.h"

struct rect {
	int x1, y1, x2, y2;
};

class Camera : public Component {
private:
	number_t _angleX;
	number_t _angleY;

	Transform* _transform;

	EngineColor* _colorMap;

	rect renderSpace;
	
	UIDrawer* _drawer;
	
	number_t _minRenderDistance;
	number_t _maxRenderDistance;

	__device__ std::pair<Vector3, Triangle> rayCastGetTriangle(Vector3 startPoint, Vector3 direction);
	__device__ EngineColor rayCast(Vector3 startPoint, Vector3 direction);
	__global__ void castRays(EngineColor* map, Vector3 xPixelVector, Vector3 yPixelVector);
	__host__ void renderColorsOnCUDA();
public:
	int typeId() override;
	Camera(GameObject* parent, UIDrawer* drawer, rect renderRect, number_t angleX = 60, number_t angleY = 40);
	Camera(GameObject* parent, UIDrawer* drawer, rect renderRect, number_t angleX = 60, number_t angleY = 40, number_t minRenderDistance = 0.1, number_t maxRenderDistance = 100);

	void awake() override;
	void update() override;
};

#endif