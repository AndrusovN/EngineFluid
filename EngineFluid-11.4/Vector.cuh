#ifndef VECTOR
#define VECTOR

#include "IMovable.cuh"
#include "ICopyable.cuh"
#include "assert.cuh"
#include <vector>

// implements dynamic array of T* with device safety (fuck Thrust with it's errors about device-undefined code)
template <typename T>
class Vector : public IMovable, public ICopyable {
private:
	T** _data;
	int _size;
	int _alloc_size;
	float _defaultFreeSpace;

public:
	__host__ Vector(const std::vector<T*>& source, float defaultFreeSpace = 1) {
		_defaultFreeSpace = defaultFreeSpace;
		_size = source.size();
		_alloc_size = _size * (1 + _defaultFreeSpace);
		_data = new T * [_alloc_size];

		for (int i = 0; i < source.size(); i++)
		{
			_data[i] = source[i];
		}
	}

	__host__ __device__ Vector(int size = 1, float defaultFreeSpace = 1) {
		_defaultFreeSpace = defaultFreeSpace;
		_size = size;
		_alloc_size = size * (1 + _defaultFreeSpace);
		_data = new T*[_alloc_size];
	}

	__host__ __device__ T*& operator [] (const int index) const {
		assert(0 <= index && index < _size);
		return _data[index];
	}

	__host__ __device__ void reshape(int size) {
		assert(size >= _size);
		T** data = new T*[size];
		for (int i = 0; i < _size; i++)
		{
			data[i] = _data[i];
		}
		delete _data;
		_data = data;
	}

	__host__ __device__ void remove(int index) {
		delete _data[index];
		for (int i = index; i < _size - 1; i++)
		{
			_data[i] = _data[i + 1];
		}
		_size--;
	}

	__host__ __device__ ICopyable* copy() override {
		Vector<T>* myCopy = new Vector<T>(_size, 0);
		myCopy->_defaultFreeSpace = _defaultFreeSpace;
		myCopy->reshape(_alloc_size);
		for (int i = 0; i < _size; i++)
		{
			T* src = nullptr;
			if (isCopyable(_data[i])) {
				src = (T*)(void*)(((ICopyable*)(void*)_data[i])->copy());
			}
			else {
				src = _data[i];
			}
			myCopy->_data[i] = src;
		}

		return myCopy;
	}

	__host__ __device__ void push(T* value) {
		if (_size == _alloc_size) {
			reshape((int)(_size * (1.0f + _defaultFreeSpace)));
		}
		_data[_size] = value;
		_size++;
	}

	__host__ __device__ int size() {
		return _size;
	}

	__host__ void moveToHost() override {
		T** data = new T*[_alloc_size];

		for (int i = 0; i < _size; i++)
		{
			T* current = (T*)malloc(sizeof(T));
			cudaMemcpy(current, _data[i], sizeof(T), cudaMemcpyDeviceToHost);
			cudaFree(_data[i]);

			if (isMovable((void*)current)) {
				((IMovable*)current)->moveToHost();
			}

			data[i] = current;
		}

		_data = data;
	}

	__host__ void moveToDevice() override {
		T** data_tmp = new  T*[_size];
		
		for (int i = 0; i < _size; i++)
		{
			T* current = nullptr;
			cudaMalloc(&current, sizeof(T));
			
			if (isMovable((void*)_data[i])) {
				((IMovable*)_data[i])->moveToDevice();
			}

			cudaMemcpy(current, _data[i], sizeof(T), cudaMemcpyHostToDevice);

			delete _data[i];

			data_tmp[i] = current;
		}

		T** data = nullptr;
		cudaMalloc(&data, sizeof(T*) * _alloc_size);
		cudaMemcpy(data, data_tmp, sizeof(T*) * _size, cudaMemcpyHostToDevice);

		delete[] data_tmp;
	}
};


#endif VECTOR