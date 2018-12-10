#include "LinearModel.hpp"
#include "../Format.hpp"

namespace CoreML {
namespace LinearModel {

    Result setOffsets(CoreML::Specification::GLMRegressor* lr, std::vector<double> offsets) {
        for(double n : offsets) {
            lr->add_offset(n);
        }
        return Result();
    }

    std::vector<double> getOffsets(const CoreML::Specification::GLMRegressor& lr) {
        std::vector<double> result;
        for(double n : lr.offset()) {
            result.push_back(n);
        }
        return result;
    }

    Result setWeights(CoreML::Specification::GLMRegressor* lr, std::vector< std::vector<double>> weights) {
        for(auto w : weights) {
            Specification::GLMRegressor::DoubleArray* cur_arr = lr->add_weights();
            for(double n : w) {
                cur_arr->add_value(n);
            }
        }
        return Result();
    }

    std::vector< std::vector<double>> getWeights(const CoreML::Specification::GLMRegressor& lr) {
        std::vector< std::vector<double>> result;
        for(auto v : lr.weights()) {
            std::vector<double> cur;
            for(double n : v.value()) {
                cur.push_back(n);
            }
            result.push_back(cur);
        }
        return result;
    }

}}