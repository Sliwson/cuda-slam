#pragma once
#include "_common.h"
#include "configuration.h"

namespace Common 
{
	using CpuCloud = std::vector<Point_f>;
	using SlamFunc = std::function <std::pair<glm::mat3, glm::vec3>(const CpuCloud&, const CpuCloud&, Configuration, int*)>;

	class TestRunner
	{
	public:

		TestRunner(SlamFunc func, std::string file = "");
		~TestRunner();

		TestRunner(const TestRunner&) = delete;
		TestRunner& operator=(const TestRunner&) = delete;

		void AddTest(Configuration config) { tests.push(config); }
		void RunAll();
		void RunSingle(Configuration config);

	private:

		int currentTestIndex = 0;

		std::string outputFile;
		FILE* fileHandle = nullptr;

		std::queue<Configuration> tests;
		SlamFunc computeFunction;
	};
}
