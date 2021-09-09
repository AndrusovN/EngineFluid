#include "Triangle.cuh"
#include <assert.h>

const bool Triangle::isInsideTriangle(Vector3 point) const
{
	return ((_vertices[1] - _vertices[0]).angle_sin(point - _vertices[0]) >= 0) &&
		((_vertices[2] - _vertices[1]).angle_sin(point - _vertices[1]) >= 0) &&
		((_vertices[0] - _vertices[2]).angle_sin(point - _vertices[2]) >= 0);
}

Triangle::Triangle(Vector3 a, Vector3 b, Vector3 c) :
	_vertices{ a, b, c },
	_center((a + b + c) / 3),
	_normal((b - a).cross(b - c)) {}

const Vector3 Triangle::rayIntersection(const Vector3& startPoint, const Vector3& direction) const
{
	// formula from https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
	number_t d = (_vertices[0] - startPoint).dot(_normal) / (direction.dot(_normal));
	Vector3 point = direction * d + startPoint;
	if (isInsideTriangle(point)) {
		return point;
	}
	else {
		return Vector3::INFINITY;
	}
}

const bool Triangle::isInside(const Vector3& other) const
{
	return _normal.dot(other - _center) <= 0;
}

const Vector3 Triangle::normal() const
{
	return _normal;
}

const Triangle Triangle::reversed() const
{
	return Triangle(_vertices[0], _vertices[2], _vertices[1]);
}

const Vector3 Triangle::get_vertex(int index) const
{
	assert(0 <= index && index <= 3);
	return _vertices[index];
}