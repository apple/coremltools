#ifndef MLMODEL_LINEAR_MODEL_SPEC_HPP
#define MLMODEL_LINEAR_MODEL_SPEC_HPP

#include "../Result.hpp"
#include "../Model.hpp"

namespace CoreML {
namespace LinearModel {

    /**
     * Set the weights.
     *
     * @param weights Two dimensional vector of doulbes.
     * @return Result type with errors.
     */
    Result setWeights(CoreML::Specification::GLMRegressor* model, std::vector< std::vector<double>> weights);

    /**
     * Set the offsets/intercepts.
     *
     * @param offsets The offsets or intercepts
     * @return Result type with errors.
     */
    Result setOffsets(CoreML::Specification::GLMRegressor* model, std::vector<double> offsets);
    
    /**
     * Get offsets/intercepts.
     *
     * @return offsets.
     */
    std::vector<double> getOffsets(const CoreML::Specification::GLMRegressor& model);

    /**
     * Get weights.
     *
     * @return weights.
     */
    std::vector< std::vector<double>> getWeights(const CoreML::Specification::GLMRegressor& model);
}}

#endif
