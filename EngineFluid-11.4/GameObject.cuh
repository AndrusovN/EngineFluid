#ifndef GAME_OBJECT
#define GAME_OBJECT

#include "IComponent.cuh"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#define PASS #ERROR("Define this part of code")

template <typename Base, typename T>
__host__ __device__ inline bool instanceof(const T* object) {
	return dynamic_cast<const Base*>(object) != nullptr;
}

class GameObject : public IMovable {
private:
	thrust::host_vector<IComponent*> _componentsHost;
	thrust::device_vector<IComponent*> _componentsDevice;

	bool _isOnDevice = false;
public:
	template <typename Type>
	__host__ __device__ Type* getComponentOfType() {
		if (_isOnDevice) {
			for (int i = 0; i < _componentsDevice.size(); i++)
			{
				auto& component = _componentsDevice[i];
				if (instanceof<Type>(component)) {
					return (Type*)component;
				}
			}
		}
		else {
			for (int i = 0; i < _componentsHost.size(); i++)
			{
				auto& component = _componentsHost[i];
				if (instanceof<Type>(component)) {
					return (Type*)component;
				}
			}
		}
		
		return nullptr;
	}

	template <typename Type>
	__host__ __device__ thrust::host_vector<Type*> getComponentsOfTypeHost() {
		thrust::host_vector<Type*> result;
		for (int i = 0; i < _componentsHost.size(); i++)
		{
			auto component = _componentsHost[i];
			if (instanceof<Type>(component)) {
				result.push_back(component);
			}
		}

		return result;
	}

	template <typename Type>
	__host__ __device__ thrust::device_vector<Type*> getComponentsOfTypeDevice() {
		thrust::device_vector<Type*> result;
		for (int i = 0; i < _componentsDevice.size(); i++)
		{
			auto component = _componentsDevice[i];
			if (instanceof<Type>(component)) {
				result.push_back(component);
			}
		}

		return result;
	}

	__host__ __device__ bool isNull() {
		return this == nullptr;
	}

	__host__ __device__ void addComponent(IComponent* component);

	__host__ __device__ void removeComponent(IComponent* component);

	__host__ __device__ thrust::device_vector<IComponent*> getComponentsDevice();
	__host__ __device__ thrust::host_vector<IComponent*> getComponentsHost();

	__host__ void moveToHost() override;
	__host__ void moveToDevice() override;
};

#endif
