#include "configuration.h"

void Common::Configuration::Print()
{
	const auto computationMethodString = [c = this->ComputationMethod]() {
		switch (c)
		{
		case ComputationMethod::Icp:
			return "Icp";
		case ComputationMethod::Cpd:
			return "Cpd";
		case ComputationMethod::NoniterativeIcp:
			return "Non iterative icp";
		default:
			return "";
		}
	}();

	const auto executionPolicyString = [e = this->ExecutionPolicy]() {
		if (!e.has_value())
			return "";

		switch (e.value())
		{
		case ExecutionPolicy::Parallel:
			return "Parallel";
		case ExecutionPolicy::Sequential:
			return "Sequential";
		default:
			return "";
		}
	}();

	printf("===============================\n");
	printf("Cuda-slam run configuration:\n");
	printf("Computation method: %s\n", computationMethodString);
	printf("Before path: %s\n", BeforePath.c_str());
	printf("After path: %s\n", AfterPath.c_str());

	if (ExecutionPolicy.has_value())
		printf("Execution policy: %s\n", executionPolicyString);

	if (Transformation.has_value())
	{
		printf("Rotation matrix:\n");
		PrintMatrix(Transformation.value().first);
		printf("Translation vector:\n");
		auto vec = Transformation.value().second;
		printf("%f, %f, %f\n", vec.x, vec.y, vec.z);
	}

	if (TransformationParameters.has_value())
	{
		printf("Rotation range: %f\n", TransformationParameters.value().first);
		printf("Translation range: %f\n", TransformationParameters.value().second);
	}

	if (MaxIterations.has_value())
		printf("Max iterations: %d\n", MaxIterations.value());

	printf("===============================\n");
}
