#include "Camera.cuh"
#include "GameManager.cuh"
#include "Mesh.cuh"
#include "Light.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assert.cuh"

__device__ Pair<Vector3, Triangle> rayCastGetTriangle(Vector3 startPoint, Vector3 direction, Scene* scene)
{
	Scene* current = scene;

	Triangle empty = Triangle(Vector3::ZERO(), Vector3::ZERO(), Vector3::ZERO());

	Vector3 nearestPoint = Vector3::INFINITY_VECTOR();
	Triangle nearestTriangle = Triangle(Vector3::ZERO(), Vector3::ZERO(), Vector3::ZERO());

	Vector<Mesh>* meshes = current->getComponents<Mesh>();

	for (int i = 0; i < meshes->size(); i++)
	{
		Mesh* mesh = (*meshes)[i];
		for (int triangleIndex = 0; triangleIndex < mesh->size(); triangleIndex++)
		{
			Triangle t = mesh->get_triangle(triangleIndex);

			Vector3 intersection = t.rayIntersection(startPoint, direction);

			if (Vector3::sqrDistance(intersection, startPoint) < Vector3::sqrDistance(nearestPoint, startPoint)) {
				nearestPoint = intersection;
				nearestTriangle = t;
			}
		}
	}

	delete meshes;

	return { nearestPoint, nearestTriangle };
}

__device__ EngineColor rayCast(Vector3 startPoint, Vector3 direction, Scene* scene)
{
	Scene* current = scene;

	Triangle empty = Triangle(Vector3::ZERO(), Vector3::ZERO(), Vector3::ZERO());

	auto pointNtriangle = rayCastGetTriangle(startPoint, direction, scene);

	Vector3 nearestPoint = pointNtriangle.first;
	Triangle nearestTriangle = pointNtriangle.second;

	if (nearestTriangle == empty) {
		return EngineColor(0, 0, 0);
	}

	EngineColor result = EngineColor(0, 0, 0, 1);

	Vector<GeneralLight>* lights = current->getComponents<GeneralLight>();

	for (int i = 0; i < lights->size(); i++)
	{
		GeneralLight* light = (*lights)[i];
		Vector3 lightPosition = light->gameObject()->getComponentOfType<Transform>()->position();

		if (nearestTriangle.normal().angle_cos(lightPosition - nearestPoint) > 0) {
			auto pointNTriangleLight = rayCastGetTriangle(nearestPoint, lightPosition - nearestPoint, scene);
			if (Vector3::sqrDistance(nearestPoint, lightPosition) < Vector3::sqrDistance(nearestPoint, pointNTriangleLight.first)) {
				result = result + light->getLight(nearestTriangle.normal());
			}
		}
	}

	delete lights;

	return result;
}

__global__ void castRays(EngineColor* map, Vector3 xPixelVector, Vector3 yPixelVector, 
	Vector3 position, Vector3 forward, number_t minRenderDistance, number_t maxRenderDistance, Scene* scene)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	int width = blockDim.y;
	int height = blockDim.x;

	// center the zero point
	x -= width / 2;
	y -= height / 2;
	y = -y;

	
	Vector3 direction = forward + xPixelVector * x + yPixelVector * y;
	Vector3 startPoint = position + direction * minRenderDistance;

	map[x * height + y] = rayCast(startPoint, direction, scene);
}

__host__ void Camera::renderColorsOnCUDA()
{
	int width = renderSpace.x2 - renderSpace.x1;
	int height = renderSpace.y2 - renderSpace.y1;

	EngineColor* device_map;
	cudaMalloc((void**)&device_map, height * width * sizeof(EngineColor));

	Vector3 xPixel = _transform->right() * tanf(_angleX / 2) * 2;
	Vector3 yPixel = _transform->up() * tanf(_angleY / 2) * 2;


	castRays<<< width, height >>>(device_map, xPixel, yPixel, _transform->position(), _transform->forward(), 
		_minRenderDistance, _maxRenderDistance, GameManager::instance()->getCurrentScene());

	EngineColor* map = new EngineColor[width * height];
	cudaMemcpy(map, device_map, height * width* sizeof(EngineColor), cudaMemcpyDeviceToHost);

	Color* total_map = new Color[ _drawer->getHeight() * _drawer->getWidth() ];
	Color* oldMap = _drawer->getColorMap();
	for (int i = 0; i < _drawer->getWidth(); i++)
	{
		for (int j = 0; j < _drawer->getHeight(); j++)
		{
			int x = i - renderSpace.x1;
			int y = j - renderSpace.y1;
			if (0 <= x && x < width && 0 <= y && y < height) {
				total_map[i * _drawer->getHeight() + j] = map[x * height + y].toWinColor();
			}
			else {
				total_map[i * _drawer->getHeight() + j] = oldMap[i * _drawer->getHeight() + j];
			}
		}
	}

	_drawer->resetColorMap(total_map);
}

__host__ __device__ int Camera::typeId() const 
{
	return CAMERA_TYPE_ID;
}

__host__ __device__ Camera::Camera(GameObject* parent) : Component(parent)
{
	_drawer = nullptr;
	_angleX = 0;
	_angleY = 0;
	_minRenderDistance = 0;
	_maxRenderDistance = 1;
	_transform = nullptr;
}

__host__ __device__ Camera::Camera(GameObject* parent, UIDrawer* drawer, rect renderRect, number_t angleX, number_t angleY) : Component(parent), _drawer(drawer)
{
	assert(angleX > 0);
	assert(angleY > 0);

	renderSpace = renderRect;
	_angleX = angleX;
	_angleY = angleY;
}

__host__ __device__ Camera::Camera(GameObject* parent, 
	UIDrawer* drawer, rect renderRect, number_t angleX, number_t angleY, 
	number_t minRenderDistance, number_t maxRenderDistance) : Component(parent), _drawer(drawer)
{
	assert(angleX > 0);
	assert(angleY > 0);
	assert(minRenderDistance > 0);
	assert(maxRenderDistance > 0);

	renderSpace = renderRect;
	_angleY = angleY / 180 * PI();
	_angleX = angleX / 180 * PI();
	_minRenderDistance = minRenderDistance;
	_maxRenderDistance = maxRenderDistance;
}

__host__ __device__ void Camera::awake()
{
	_transform = gameObject()->getComponentOfType<Transform>();
	assert(_transform != nullptr);
}

__host__ void Camera::deviceUpdate()
{
	renderColorsOnCUDA();
}

__host__ void Camera::moveToDevice()
{
}

__host__ void Camera::moveToHost()
{
}

__host__ __device__ void Camera::resetGameObject(GameObject* object)
{
	Component::resetGameObject(object);
	_transform = object->getComponentOfType<Transform>();
}
