#include "ImportMesh.h"


std::string cut(std::string base, int ofset) {
	if (ofset > 0) {
		std::string res = "";
		for (int i = 0; i < ofset; i++)
		{
			res += base[i];
		}
		return res;
	}
	else
	{
		if (ofset < 0) {
			std::string res;
			for (int i = base.length() + ofset; i < base.length(); i++)
			{
				res += base[i];
			}
			return res;
		}
		else
		{
			return(base);
		}
	}
}

std::vector<Vector3> readVertices(std::ifstream& file, std::string& data) {
	std::vector<Vector3> vertices;
	int num = 0;
	Vector3 vertice;
	while (cut(data, -2) != "vn")
	{
		file >> data;
		std::string _data;
		_data = data;
		if (data[data.length() - 1] == 'v') {
			continue;
		}

		if (data[data.length() - 2] == 'v') {
			break;
			_data = cut(data, data.length() - 2);
		}

		double value = stof(_data);

		switch (num)
		{
		case 0:
			vertice.set_x(value);
			break;
		case 1:
			vertice.set_y(value);
			break;
		case 2:
			vertice.set_z(value);
			vertices.push_back(vertice);
			num = -1;
			break;
		default:
			break;
		}
		num++;
	}

	return vertices;
}

std::vector<Vector3> readNormals(std::ifstream& file, std::string& data) {
	std::vector<Vector3> normals;
	Vector3 vertice;
	int num;

	Vector3 normal;

	while (cut(data, -6) != "usemtl")
	{
		file >> data;
		std::string _data;
		_data = data;
		if (data[data.length() - 6] == 'u') {
			break;
			_data = cut(data, data.length() - 6);
		}

		if (data[data.length() - 2] == 'v') {
			continue;
			_data = cut(data, data.length() - 2);
		}

		double value = stof(_data);

		switch (num)
		{
		case 0:
			normal.set_x(value);
			break;
		case 1:
			normal.set_y(value);
			break;
		case 2:
			normal.set_z(value);
			normals.push_back(normal);
			num = -1;
			break;
		default:
			break;
		}
		num++;
	}

	return normals;
}

std::vector<Face> readFaces(std::ifstream& file) {
	std::vector<Face> faces;
	Face face = Face();
	std::string data;

	while (!file.eof()) {
		file >> data;
		if (data[data.length() - 1] == 'f') {
			continue;
			data = cut(data, data.length() - 1);
		}
		int i;
		for (i = 0; i < data.length(); i++)
		{
			if (data[i] == '/') {
				break;
			}
		}
		//int j

		std::string _data = cut(data, i);
		std::string _data_ = cut(data, i + 2 - data.length());

		int res = stoi(_data);
		int group = stoi(_data_);
		if (group - 1 != face.normalNumber) {
			faces.push_back(face);
			face.verticesNumbers = std::vector<int>();
			face.normalNumber = group - 1;
		}
		face.verticesNumbers.push_back(res - 1);
	}

	faces.push_back(face);

	return faces;
}

std::vector<Triangle> prepareTriangles(std::vector<Vector3> vertices, std::vector<Vector3> normals, std::vector<Face> faces) {
	//cout << "faces length: " << faces.length() << endl;
	std::vector<Triangle> triangles = std::vector<Triangle>();
	for (int i = 0; i < faces.size(); i++)
	{
		//cout << i << " : " << faces.get(i).verticesNumbers.length() << endl;
		for (int j = 1; j < faces[i].verticesNumbers.size() - 1; j++)
		{
			Vector3 a = vertices[faces[i].verticesNumbers[0]];
			Vector3 b = vertices[faces[i].verticesNumbers[j]];
			Vector3 c = vertices[faces[i].verticesNumbers[j + 1]];

			Triangle tr = Triangle(a, b, c);

			if (tr.normal().angle_cos(normals[faces[i].normalNumber]) < 0) {
				tr = tr.reversed();
			}

			triangles.push_back(tr);
		}
	}
	return triangles;
}

std::vector<Triangle> importAssetMesh(std::string path) {
	std::ifstream file;
	file.open(path, std::ios_base::in);

	std::string data = "qwerty";

	while (data.length() > 0 || data[data.length() - 1] != 'v') {
		file >> data;
	}

	std::vector<Vector3> vertices = readVertices(file, data);
	std::vector<Vector3> normals = readNormals(file, data);

	while (data != "f") {
		file >> data;
	}

	std::vector<Face> faces = readFaces(file);

	return prepareTriangles(vertices, normals, faces);
}