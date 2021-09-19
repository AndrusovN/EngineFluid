#ifndef LIGHT
#define LIGHT

#include "Component.cuh"
#include "Transform.cuh"
#include "Drawing.h"

typedef char _byte;

const int GENERAL_LIGHT_TYPEID = 2;

struct EngineColor {
	_byte r, g, b, a;

	EngineColor();
	EngineColor(_byte r, _byte g, _byte b, _byte a = 1);
	EngineColor(Color base);

	EngineColor operator = (const EngineColor& other);
	const EngineColor operator + (const EngineColor& other) const;
	const EngineColor operator == (const EngineColor& other) const;

	Color toWinColor();
};

class GeneralLight : public Component {
private:
	Transform* _transform;
	EngineColor _lightColor;
public:
	__host__ __device__ const int typeId() const override;

	GeneralLight(GameObject* parent, EngineColor lightColor = EngineColor(255, 255, 255));

	__host__ __device__ void awake() override;

	__host__ __device__ EngineColor getLight(Vector3 normal);

	__host__ void moveToDevice() override;
	__host__ void moveToHost() override;

	__host__ __device__ void resetGameObject(GameObject* object) override;
};

#endif
