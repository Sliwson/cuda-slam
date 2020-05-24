#pragma once

#include "_common.h"

namespace Common {
	struct StageProperties
	{
		bool IsRunning = false;
		std::chrono::time_point<std::chrono::high_resolution_clock> Begin;
		std::chrono::milliseconds MilisecondsElpased = std::chrono::duration_values<std::chrono::milliseconds>::zero();
	};

	class Timer {
	public:
		Timer(const std::string& name = "Timer") : timerName(name) {}
		Timer(const Timer&) = delete;
		Timer& operator=(const Timer&) = delete;

		std::shared_ptr<StageProperties> AddStage(const std::string& name);
		void StartStage(const std::string& name);
		void StopStage(const std::string& name);
		void StageTimedCall(const std::string& name, std::function<void()> func);

		void PrintResults();

	private:

		std::shared_ptr<StageProperties> GetStage(const std::string& name);
		std::map<std::string, std::shared_ptr<StageProperties>> stages;
		std::string timerName = "";

	};
}
