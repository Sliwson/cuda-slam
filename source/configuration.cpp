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

	printf("===============================\n");
	printf("Cuda-slam run configuration:\n");
	printf("Computation method: %s\n", computationMethodString);
	printf("Before path: %s\n", BeforePath.c_str());
	printf("After path: %s\n", AfterPath.c_str());
	printf("===============================\n");
}
