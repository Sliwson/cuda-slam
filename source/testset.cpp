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
    }

    std::vector<Configuration> Common::GetBasicTestSet()
    {
        std::vector<Configuration> set;
        for (int i = 0; i < 20; i++)
        {
            auto config = GetDefaultConfiguration();
            config.TransformationParameters = std::make_pair(0.05f * i, 10.f);
            set.push_back(config);
        }

        return set;
    }
}
