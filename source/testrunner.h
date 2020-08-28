#pragma once
#include "_common.h"
#include "configuration.h"

namespace Common 
{
	using cpu_cloud = std::vector<Point_f>;
	using slam_func = std::function <std::pair<glm::mat3, glm::vec3>(const cpu_cloud&, const cpu_cloud&, Configuration)>;

	class TestRunner
	{
	public:

		TestRunner(slam_func func) : computeFunction(func) {}
		TestRunner(const TestRunner&) = delete;
		TestRunner& operator=(const TestRunner&) = delete;

		void AddTest(Configuration config) { tests.push(config); }
		void RunAll();
		void RunSingle(Configuration config);

	private:

		std::queue<Configuration> tests;
		slam_func computeFunction;
	};
}
