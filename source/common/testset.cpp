#include "testset.h"
#include "configuration.h"

namespace Common
{
    namespace {
        Configuration GetDefaultConfiguration()
        {
            Configuration config;
            config.BeforePath = "data/bunny.obj";
            config.AfterPath = "data/bunny.obj";
            config.ComputationMethod = ComputationMethod::Icp;
            config.MaxIterations = 50;
            config.MaxDistanceSquared = 10000.f;
            config.TransformationParameters = std::make_pair(0.3f, 10.f);
            return config;
        }

        std::string GetObjectWithMinSize(int size)
        {
            const auto mainName = [size]() {
                if (size <= 14904)
                    return "bunny";
                else if (size <= 35008)
                    return "bird";
                else if (size <= 333536)
                    return "rose";
                else if (size <= 376401)
                    return "mustang";
                else if (size <= 1375028)
                    return "airbus";
                else
                    assert(false);
                    return "";
            }();

            return "data/" + std::string(mainName) + ".obj";
        }

        struct MethodTestParams
        {
			int MinSize = 500;
			int SizeSpan = 500;
			int MaxSize = 100000;
        };
    }

    std::vector<Configuration> GetSizesTestSet(ComputationMethod method)
    {
        const std::map<ComputationMethod, MethodTestParams> map{ {
            { ComputationMethod::Icp, { 1000, 4000, 100000 }},
            { ComputationMethod::Cpd, { 100, 100, 1000 }},
            { ComputationMethod::NoniterativeIcp, { 1000, 4000, 200000 }}
        } };

        std::vector<Configuration> configurations;

        const auto params = map.find(method)->second;
        for (int i = params.MinSize; i <= params.MaxSize; i += params.SizeSpan)
        {
            auto path = GetObjectWithMinSize(i);

			Configuration config;
            config.BeforePath = path;
            config.AfterPath = path;
            config.ComputationMethod = method;
            config.MaxIterations = 50;
            config.MaxDistanceSquared = 10000.f;
            config.TransformationParameters = std::make_pair(.2f, 10.f);
            config.CloudBeforeResize = i;
            config.CloudAfterResize = i;
            config.ExecutionPolicy = method == ComputationMethod::Icp ? ExecutionPolicy::Parallel : ExecutionPolicy::Sequential;
            config.ApproximationType = ApproximationType::None;
            config.CpdWeight = 0.1f;

            configurations.push_back(config);
        }

        return configurations;
	}

    std::vector<Configuration> GetPerformanceTestSet(ComputationMethod method)
    {
        const std::map<ComputationMethod, MethodTestParams> map{ {
            { ComputationMethod::Icp, { 25000, 25000, 1300000 }},
            { ComputationMethod::Cpd, { 100, 100, 1000 }},
            { ComputationMethod::NoniterativeIcp, { 10000, 10000, 300000 }}
        } };

        std::vector<Configuration> configurations;

        const auto params = map.find(method)->second;
        for (int i = params.MinSize; i <= params.MaxSize; i += params.SizeSpan)
        {
            auto path = GetObjectWithMinSize(i);

			Configuration config;
            config.BeforePath = path;
            config.AfterPath = path;
            config.ComputationMethod = method;
            config.MaxIterations = 50;
            config.CloudSpread = 10.f;
            config.MaxDistanceSquared = 10000.f;
            config.TransformationParameters = std::make_pair(.2f, 10.f);
            config.CloudBeforeResize = i;
            config.CloudAfterResize = i;
            config.ExecutionPolicy = ExecutionPolicy::Sequential;
            config.ApproximationType = ApproximationType::Hybrid;
            config.NicpSubcloudSize = 1000;
            config.NicpIterations = 64;
            config.CpdWeight = 0.1f;

            configurations.push_back(config);
        }

        return configurations;
    }

    std::vector<Configuration> GetConvergenceTestSet(ComputationMethod method)
    {
        const std::map<ComputationMethod, MethodTestParams> mapCpu{ {
           { ComputationMethod::Icp, { 20000, 20000, 100000 }},
           { ComputationMethod::Cpd, { 4000, 4000, 20000 }},
           { ComputationMethod::NoniterativeIcp, { 250000, 250000, 1250000 }}
       } };

        const std::map<ComputationMethod, MethodTestParams> mapGpu{ {
           { ComputationMethod::Icp, { 20000, 20000, 100000 }},
           { ComputationMethod::Cpd, { 4000, 4000, 20000 }},
           { ComputationMethod::NoniterativeIcp, { 250000, 250000, 1250000 }}
       } };

        std::vector<Configuration> configurations;

        const auto params = mapCpu.find(method)->second;
        for (int j = 0; j < 10; j++)
        {
            for (int i = params.MinSize; i <= params.MaxSize; i += params.SizeSpan)
            {
                auto path = GetObjectWithMinSize(i);

                // Default config
                Configuration config;
                config.BeforePath = path;
                config.AfterPath = path;
                config.ComputationMethod = method;
                config.MaxIterations = 100;
                config.CloudSpread = 10.f;
                config.MaxDistanceSquared = 10000.f;
                config.TransformationParameters = std::make_pair(.2f, 10.f);
                config.CloudBeforeResize = i;
                config.CloudAfterResize = i;
                config.ExecutionPolicy = ExecutionPolicy::Parallel;
                config.ApproximationType = method == ComputationMethod::Cpd ? ApproximationType::Hybrid : ApproximationType::None;
                config.NicpSubcloudSize = 5000;
                config.NicpBatchSize = 1;
                config.NicpIterations = 16;
                config.CpdWeight = 0.1f;
                config.CpdTolerance = 1e-4;

                configurations.push_back(config);

                // Same translation
                config.TransformationParameters = std::make_pair(.4f, 10.f);
                configurations.push_back(config);

                config.TransformationParameters = std::make_pair(.6f, 10.f);
                configurations.push_back(config);

                // Larger translation
                config.TransformationParameters = std::make_pair(.2f, 20.f);
                configurations.push_back(config);

                config.TransformationParameters = std::make_pair(.4f, 20.f);
                configurations.push_back(config);

                config.TransformationParameters = std::make_pair(.6f, 20.f);
                configurations.push_back(config);

                // Huge translation
                config.TransformationParameters = std::make_pair(.2f, 30.f);
                configurations.push_back(config);

                config.TransformationParameters = std::make_pair(.4f, 30.f);
                configurations.push_back(config);

                config.TransformationParameters = std::make_pair(.6f, 30.f);
                configurations.push_back(config);
            }
        }

        return configurations;
    }
}
