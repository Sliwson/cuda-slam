#pragma once

#include "cuda.cuh"

namespace CUDAProbabilities
{
	struct Probabilities
	{
		Probabilities(const int& cloudBeforeSize, const int& cloudAfterSize)
		{
			p1 = thrust::device_vector<float>(cloudBeforeSize);
			pt1 = thrust::device_vector<float>(cloudAfterSize);
			px = thrust::device_vector<glm::vec3>(cloudBeforeSize);
			p = thrust::device_vector<float>(cloudBeforeSize);
			tmp = thrust::device_vector<float>(cloudBeforeSize);
			error = 0.0f;
		}

		// The probability matrix, multiplied by the identity vector.
		thrust::device_vector<float> p1;
		// The probability matrix, transposed, multiplied by the identity vector.
		thrust::device_vector<float> pt1;
		// The probability matrix multiplied by the fixed(cloud before) points.
		thrust::device_vector<glm::vec3> px;
		// The total error.
		float error = 0.0f;
		//p
		thrust::device_vector<float> p;
		//tmp
		thrust::device_vector<float> tmp;		
	};
}
