#pragma once

namespace Common
{
	enum class ComputationMethod
	{
		Icp,
		NoniterativeIcp,
		Cpd
	};

	enum class ExecutionPolicy
	{
		Sequential,
		Parallel
	};

	enum class ApproximationType
	{
		None,
		Full,
		Hybrid
	};
}
