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
    }

    std::vector<Configuration> GetSizesTestSet(ComputationMethod method)
    {
        constexpr int sizeSpan = 2500;
        constexpr int minSize = 2500;
        constexpr int maxSize = 7500;

        std::vector<Configuration> configurations;

        for (int i = minSize; i <= maxSize; i += sizeSpan)
        {
            auto path = GetObjectWithMinSize(i);

			Configuration config;
            config.BeforePath = path;
            config.AfterPath = path;
            config.ComputationMethod = method;
            config.MaxIterations = 50;
            config.MaxDistanceSquared = 10000.f;
            config.TransformationParameters = std::make_pair(.5f, 10.f);
            config.CloudResize = i;
            config.ExecutionPolicy = ExecutionPolicy::Sequential;

            configurations.push_back(config);
        }

        return configurations;
	}
}
