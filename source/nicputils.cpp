#include "nicputils.h"

namespace Common
{
	void StoreResultIfOptimal(std::vector<NonIterativeSlamResult>& results, const NonIterativeSlamResult& newResult, const int desiredLength)
	{
		int length = results.size();
		if (length == 0 && desiredLength > 0)
		{
			results.push_back(newResult);
			return;
		}

		for (int i = 0; i < length; i++)
		{
			if (newResult.getApproximatedError() < results[i].getApproximatedError())
			{
				results.insert(results.begin() + i, newResult);
				if (results.size() > desiredLength)
				{
					results.resize(desiredLength);
					return;
				}
			}
		}
	}
}
