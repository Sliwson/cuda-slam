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
            config.TransformationParameters = std::make_pair(1.f, 1.f);
            return config;
        }
    }

    std::vector<Configuration> Common::GetBasicTestSet()
    {
        return {
            GetDefaultConfiguration(),
            GetDefaultConfiguration(),
            GetDefaultConfiguration(),
            GetDefaultConfiguration()
        };
    }
}
