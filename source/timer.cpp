#include "timer.h"

namespace Common
{
	std::shared_ptr<StageProperties> Timer::AddStage(const std::string& name)
	{
		auto found = GetStage(name);
		if (found != nullptr)
			return found;

		auto stage = std::make_shared<StageProperties>();
		stages.insert(std::make_pair(name, stage));
		return stage;
	}

	void Timer::StartStage(const std::string& name)
	{
		auto stage = GetStage(name);
		if (stage == nullptr)
			stage = AddStage(name);

		if (!stage->IsRunning)
		{
			stage->Begin = std::chrono::high_resolution_clock::now();
			stage->IsRunning = true;
		}
	}

	void Timer::StopStage(const std::string& name)
	{
		auto stage = GetStage(name);
		if (stage == nullptr || !stage->IsRunning)
			return;

		const auto duration = std::chrono::high_resolution_clock::now() - stage->Begin;
		const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
		stage->MilisecondsElpased += milliseconds;
		stage->IsRunning = false;
	}

	void Timer::StageTimedCall(const std::string& name, std::function<void()> func)
	{
		StartStage(name);
		func();
		StopStage(name);
	}

	void Timer::PrintResults()
	{
		if (std::any_of(stages.begin(), stages.end(), [](auto prop) { return prop.second->IsRunning; }))
			printf("One or more timers are still running!");

		printf("%s results:\n", timerName.c_str());
		for (const auto& stage : stages)
			printf("%s -> %lldms\n", stage.first.c_str(), static_cast<long long int>(stage.second->MilisecondsElpased.count()));
	}

	std::shared_ptr<StageProperties> Timer::GetStage(const std::string& name)
	{
		const auto it = stages.find(name);
		if (it != stages.end())
			return (*it).second;
		else
			return nullptr;
	}
}
