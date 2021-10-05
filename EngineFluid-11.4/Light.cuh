#ifndef LIGHT
#define LIGHT

#include "Component.cuh"
#include "Transform.cuh"
#include "Drawing.h"

typedef unsigned char _byte;

const int GENERAL_LIGHT_TYPEID = 2;

struct EngineColor {
	_byte r, g, b, a;

	__host__ __device__ EngineColor();
	__host__ __device__ EngineColor(_byte r, _byte g, _byte b, _byte a = 1);
	__host__ __device__ EngineColor(Color base);

	__host__ __device__ EngineColor operator = (const EngineColor& other);
	__host__ __device__ const EngineColor operator + (const EngineColor& other) const;
	__host__ __device__ const EngineColor operator == (const EngineColor& other) const;

	__host__ __device__ Color toWinColor();
};

class GeneralLight : public Component {
private:
	Transform* _transform;
	EngineColor _lightColor;
public:
	__host__ __device__ int typeId() const override;

	__host__ __device__ GeneralLight(GameObject* parent = nullptr, EngineColor lightColor = EngineColor(255, 255, 255));

	__host__ __device__ void awake() override;

	__host__ __device__ EngineColor getLight(Vector3 normal);

	__host__ void moveToDevice() override;
	__host__ void moveToHost() override;

	__host__ __device__ void resetGameObject(GameObject* object) override;
};

#endif
