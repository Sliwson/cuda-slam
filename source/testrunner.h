#pragma once
#include "_common.h"
#include "configuration.h"

namespace Common 
{
	using CpuCloud = std::vector<Point_f>;
	using SlamFunc = std::function <std::pair<glm::mat3, glm::vec3>(const CpuCloud&, const CpuCloud&, Configuration)>;

	class TestRunner
	{
	public:

		TestRunner(SlamFunc func) : computeFunction(func) {}
		TestRunner(const TestRunner&) = delete;
		TestRunner& operator=(const TestRunner&) = delete;

		void AddTest(Configuration config) { tests.push(config); }
		void RunAll();
		void RunSingle(Configuration config);

	private:

		std::queue<Configuration> tests;
		SlamFunc computeFunction;
	};
}
