#pragma once
#include "tsp_types.h"
#include <algorithm>
#include <cmath>

namespace tsp {

struct EliteSelector {
    enum class Scheme { CES_LogWeights, CEM_Uniform };
    Scheme scheme{Scheme::CES_LogWeights};

    // returns indices of top-k by total cost
    template<class CostFn>
    std::vector<size_t> select(const std::vector<PathCandidate>& C, double frac, CostFn cost) const {
        const int k = std::max(1, int(C.size() * frac));
        std::vector<size_t> idx(C.size());
        for (size_t i=0;i<C.size();++i) idx[i]=i;
        std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
            [&](size_t a, size_t b){ return cost(C[a]) < cost(C[b]); });
        idx.resize(k);
        return idx;
    }

    std::vector<double> weights(int k) const {
        std::vector<double> w(k, 1.0);
        if (scheme == Scheme::CEM_Uniform) return w;
        // CES log-weights
        double sumw=0.0;
        for (int i=0;i<k;++i){ w[i] = std::log(k + 0.5) - std::log(i + 1.0); sumw += w[i]; }
        for (double& wi : w) wi /= sumw;
        return w;
    }
};

} // namespace tsp
