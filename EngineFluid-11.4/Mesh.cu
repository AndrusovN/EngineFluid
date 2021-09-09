#include "Mesh.cuh"
#include <assert.h>
#include "ImportMesh.h"

Vector3 Mesh::rotatePoint(Vector3 point, Quaternion rotation)
{
	return _center + (rotation.applyToVector(point - _center));
}

Mesh::Mesh(Vector3 position, number_t scale) : Mesh((Triangle*)nullptr, 0, position, scale) {}

Mesh::Mesh(Triangle* triangles, int size, Vector3 position, number_t scale)
{
	assert(scale < -1 * EPSILON || scale > EPSILON);
	_appliedRotation = Quaternion();
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

Mesh::Mesh(const char* filename, int stringSize, Vector3 position, number_t scale)
{
	assert(scale < -1 * EPSILON || scale > EPSILON);
	_appliedRotation = Quaternion();
	
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

Triangle Mesh::get_triangle(int index)
{
	assert(0 <= index && index < _triangles_size);
	return _triangles[index];
}

int Mesh::size()
{
	return _triangles_size;
}

void Mesh::setAppliedRotation(Quaternion rotation)
{
	rotate(_appliedRotation.inversed() * rotation);
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

void Mesh::setPosition(Vector3 position)
{
	translate(position - _center);
}

bool Mesh::isPointInside(Vector3 point)
{
	bool variant = false;
	for (int i = 0; i < _triangles_size; i++)
	{
		bool currentState = _triangles[i].isInside(point);
		if (variant != 0 && currentState != variant) {
			return false;
		}

		variant = currentState;
	}

	return true;
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

