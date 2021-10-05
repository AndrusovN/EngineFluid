#ifndef CAMERA
#define CAMERA

#include "Component.cuh"
#include "Light.cuh"
#include "Scene.cuh"
#include "Triangle.cuh"
#include "Transform.cuh"
#include "Drawing.h"
#include "Pair.cuh"

const int CAMERA_TYPE_ID = 1;

struct rect {
	int x1, y1, x2, y2;
};


__device__ Pair<Vector3, Triangle> rayCastGetTriangle(Vector3 startPoint, Vector3 direction, Scene* scene);
__device__ EngineColor rayCast(Vector3 startPoint, Vector3 direction, Scene* scene);
__global__ void castRays(EngineColor* map, Vector3 xPixelVector, Vector3 yPixelVector, 
	Vector3 position, Vector3 forward, number_t minRenderDistance, number_t maxRenderDistance, Scene* scene);

class Camera : public Component {
private:
	number_t _angleX;
	number_t _angleY;

	Transform* _transform;

	rect renderSpace;
	
	UIDrawer* _drawer;
	
	number_t _minRenderDistance;
	number_t _maxRenderDistance;

	__host__ void renderColorsOnCUDA();
public:
	__host__ __device__ int typeId() const override;
	__host__ __device__ Camera(GameObject* parent = nullptr);
	__host__ __device__ Camera(GameObject* parent, UIDrawer* drawer, rect renderRect, number_t angleX = 60, number_t angleY = 40);
	__host__ __device__ Camera(GameObject* parent, UIDrawer* drawer, rect renderRect, number_t angleX = 60, number_t angleY = 40, 
		number_t minRenderDistance = 0.1, number_t maxRenderDistance = 100);

	__host__ __device__ void awake() override;
	__host__ void deviceUpdate() override;
	__host__ void moveToDevice() override;
	__host__ void moveToHost() override;

	__host__ __device__ void resetGameObject(GameObject* object) override;
};

#endif