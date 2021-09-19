#include "Mesh.cuh"
#include "assert.cuh"
#include "ImportMesh.h"

Vector3 Mesh::rotatePoint(Vector3 point, Quaternion rotation)
{
	return _center + (rotation.applyToVector(point - _center));
}

int Mesh::countIntersections(Vector3 from, Vector3 direction)
{
	int result = 0;
	for (int i = 0; i < _triangles_size; i++)
	{
		Vector3 intersection = _triangles[i].rayIntersection(from, direction);
		if (intersection != Vector3::INFINITY_VECTOR) {
			result++;
		}
	}
}

Mesh::Mesh(GameObject* parent, Vector3 position, number_t scale) : Mesh(parent, (Triangle*)nullptr, 0, position, scale) {}

Mesh::Mesh(GameObject* parent, Triangle* triangles, int size, Vector3 position, number_t scale) : Component(parent)
{
	assert(scale < -1 * EPSILON || scale > EPSILON);
	_triangles = new Triangle[size];

	for (int i = 0; i < size; i++)
	{
		_triangles[i] = triangles[i];
	}

	_center = Vector3::ZERO;
	translate(position);
	_scale = 1;
	setScale(scale);
}

Mesh::Mesh(GameObject* parent, const char* filename, int stringSize, Vector3 position, number_t scale) : Component(parent)
{
	assert(scale < -1 * EPSILON || scale > EPSILON);
	
	std::vector<Triangle> triangles = importAssetMesh(std::string(filename, stringSize));

	_triangles = new Triangle[triangles.size()];

	for (int i = 0; i < triangles.size(); i++)
	{
		_triangles[i] = triangles[i];
	}

	_triangles_size = triangles.size();

	_center = Vector3::ZERO;
	translate(position);
	_scale = 1;
	this->scale(scale);
}

void Mesh::moveToCUDA()
{
	Triangle* cudaTriangles = nullptr;
	cudaMalloc((void**)&cudaTriangles, _triangles_size * sizeof(Triangle));
	cudaMemcpy(_triangles, cudaTriangles, _triangles_size * sizeof(Triangle), cudaMemcpyHostToDevice);

	delete[] _triangles;
	_triangles = cudaTriangles;
}

const int Mesh::typeId() const
{
	return MESH_TYPE_ID;
}

Triangle Mesh::get_triangle(int index)
{
	assert(0 <= index && index < _triangles_size);
	return _triangles[index];
}

int Mesh::size()
{
	return _triangles_size;
}

void Mesh::rotate(Quaternion rotation)
{
	for (int i = 0; i < _triangles_size; i++)
	{
		Vector3 a = rotatePoint(_triangles[i].get_vertex(0), rotation);
		Vector3 b = rotatePoint(_triangles[i].get_vertex(1), rotation);
		Vector3 c = rotatePoint(_triangles[i].get_vertex(2), rotation);

		_triangles[i] = Triangle(a, b, c);
	}
}

void Mesh::translate(Vector3 offset)
{
	for (int i = 0; i < _triangles_size; i++)
	{
		Vector3 a = _triangles[i].get_vertex(0) + offset;
		Vector3 b = _triangles[i].get_vertex(1) + offset;
		Vector3 c = _triangles[i].get_vertex(2) + offset;

		_triangles[i] = Triangle(a, b, c);
	}
}

bool Mesh::isPointInside(Vector3 point)
{
	Vector3 firstTry = Vector3::UP;
	int intersectionNumber = countIntersections(point, firstTry);
	if (intersectionNumber % 2 == 1) {
		return true;
	}

	Vector3 secondTry = Vector3::RIGHT;
	intersectionNumber = countIntersections(point, secondTry);
	return (intersectionNumber % 2 == 1);
}

void Mesh::setScale(number_t scale)
{
	assert(scale < -1 * EPSILON || scale > EPSILON);
	this->scale(scale / _scale);
}

void Mesh::scale(number_t times)
{
	for (int i = 0; i < _triangles_size; i++)
	{
		Vector3 a = _triangles[i].get_vertex(0) * times;
		Vector3 b = _triangles[i].get_vertex(1) * times;
		Vector3 c = _triangles[i].get_vertex(2) * times;

		_triangles[i] = Triangle(a, b, c);
	}
}

number_t Mesh::getScale()
{
	return _scale;
}

void Mesh::moveToDevice()
{
	moveToCUDA();
}

void Mesh::moveToHost()
{
	Triangle* hostTriangles = new Triangle[_triangles_size];
	cudaMemcpy(_triangles, hostTriangles, _triangles_size * sizeof(Triangle), cudaMemcpyDeviceToHost);
	cudaFree(_triangles);
	_triangles = hostTriangles;
}

