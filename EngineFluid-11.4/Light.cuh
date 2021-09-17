#ifndef LIGHT
#define LIGHT

#include "Component.cuh"
#include "Transform.cuh"
#include "Drawing.h"

typedef char byte;

struct EngineColor {
	byte r, g, b, a;

	EngineColor();
	EngineColor(byte r, byte g, byte b, byte a = 1);
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
	int typeId() override;

	GeneralLight(GameObject* parent, EngineColor lightColor = EngineColor(255, 255, 255));

	__host__ __device__ void awake() override;

	__host__ __device__ EngineColor getLight(Vector3 normal);
};

#endif
