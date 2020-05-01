#include "timer.h"

namespace Common
{
	std::shared_ptr<StageProperties> Timer::AddStage(const std::string& name)
	{
		if (auto foundStage = GetStage(name))
			return foundStage;

		auto stage = std::make_shared<StageProperties>(name);
		stages.push_back(stage);
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
		if (std::any_of(stages.begin(), stages.end(), [](auto prop) { return prop->IsRunning; }))
			printf("One or more timers are still running!");

		printf("%s results:\n", timerName.c_str());
		for (const auto& stage : stages)
			printf("%s -> %lldms\n", stage->Name.c_str(), static_cast<long long int>(stage->MilisecondsElpased.count()));
	}

	std::shared_ptr<StageProperties> Timer::GetStage(const std::string& name)
	{
		auto it = std::find_if(stages.begin(), stages.end(), [&name](auto prop) { return prop->Name == name; });
		if (it != stages.end())
			return *it;
		else
			return nullptr;
	}
}
