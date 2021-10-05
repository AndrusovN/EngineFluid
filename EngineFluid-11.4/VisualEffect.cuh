#ifndef VISUAL_EFFECT
#define VISUAL_EFFECT

#include "Drawing.h"
#include "Scene.cuh"

class VisualEffect {
public:
	__host__ __device__ virtual void apply (Color* screen, int width, int height, Scene* scene) = 0;
};

#endif