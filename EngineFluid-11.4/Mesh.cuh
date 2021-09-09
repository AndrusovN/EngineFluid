#ifndef MESH
#define MESH
#include "Triangle.cuh"
#include "Quaternion.cuh"

class Mesh {
private:
	Triangle* _triangles;
	int _triangles_size = 0;
	Vector3 _center;
	__host__ __device__ void rotatePoint(Vector3 point, Quaternion rotation);
public:
	Mesh();
};

#endif