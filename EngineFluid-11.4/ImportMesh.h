#ifndef IMPORT_MESH
#define IMPORT_MESH

#include "Triangle.cuh"
#include <string>
#include <fstream>
#include <vector>

// I wrote this cringe 2 years ago, so actually I hardly understand what happens here...
// But this somehow imports Mesh from .OBJ file
// .OBJ file is stored as set of points and set of faces (each face connects some points and defines it's normal)
// But face can store more than 3 points - it can  be any plane polygon

// A struct for .obj face
struct Face {
	std::vector<int> verticesNumbers;
	int normalNumber;

	Face() {
		verticesNumbers = std::vector<int>();
		normalNumber = 0;
	}
};

// Cuts the string
// if offset is positive - returns first <offset> symbols
// if offset is negative - returns last <-offset> symbols
// if offset is 0 - returns the whole string
std::string cut(std::string base, int offset);

// reads vertices from file (should be called only in order as in function importAssetMesh)
std::vector<Vector3> readVertices(std::ifstream& file, std::string& data);

// reads normals from file (should be called only in order as in function importAssetMesh)
std::vector<Vector3> readNormals(std::ifstream& file, std::string& data);

// reads faces from file (should be called only in order as in function importAssetMesh)
std::vector<Face> readFaces(std::ifstream& file);

// makes list of Triangles by given faces, vertices and normals
std::vector<Triangle> prepareTriangles(std::vector<Vector3> vertices, std::vector<Vector3> normals, std::vector<Face> faces);

// the main function - imports mesh from .OBJ file described by path
std::vector<Triangle> importAssetMesh(std::string path);

#endif