#ifndef MESH
#define MESH
#include "Triangle.cuh"
#include "Quaternion.cuh"
#include "Component.cuh"

const int MESH_TYPE_ID = 3;

class Mesh : public Component {
private:
	Triangle* _triangles;
	int _triangles_size = 0;
	Vector3 _center = Vector3::ZERO();
	number_t _scale = 1;

	__host__ __device__ Vector3 rotatePoint(Vector3 point, Quaternion rotation);
	__host__ __device__ int countIntersections(Vector3 from, Vector3 direction);
public:
	__host__ __device__ Mesh(GameObject* parent = nullptr);
	__host__ __device__ Mesh(GameObject* parent, Vector3 position, number_t scale = 1);
	__host__ __device__ Mesh(GameObject* parent, Triangle* triangles, int size, Vector3 position, number_t scale = 1);
	__host__ Mesh(GameObject* parent, const char* filename, int stringSize, Vector3 position, number_t scale = 1);

	__host__ void moveToCUDA();

	__host__ __device__ int typeId() const override;

	__host__ __device__ Triangle get_triangle(int index);

	__host__ __device__ int size();

	__host__ __device__ void rotate(Quaternion rotation);

	__host__ __device__ void translate(Vector3 offset);

	__host__ __device__ bool isPointInside(Vector3 point);

	__host__ __device__ void setScale(number_t scale);

	__host__ __device__ void scale(number_t times);

	__host__ __device__ number_t getScale();

	__host__ void moveToDevice() override;
	__host__ void moveToHost() override;
};

#endif