//
// Created by hrens on 5/4/18.
//

#ifndef PBRT_V3_STORM_SDGUIDED_H
#define PBRT_V3_STORM_SDGUIDED_H

#include "pbrt.h"
#include "integrator.h"
#include "lightdistrib.h"
#include "scene.h"
#include "sampler.h"
#include "camera.h"
#include "sdtree.h"


namespace pbrt {


///
/// Implementation of the Practical Path Guiding from Müller and al., 2017.
/// It is similar to the version described in the paper, implemented on top
/// of a unidirectional path tracer.
///
class SDGuidedIntegrator : public Integrator {
protected:
    /// The informations required to backpropagate the light informations to
    /// store into the recording tree.
    struct PathVertex {
        Point3f p;
        Vector3f wi;
        Spectrum beta;
    };

public:
    SDGuidedIntegrator(int maxDepth,
                       std::shared_ptr<const Camera> camera,
                       std::shared_ptr<Sampler> sampler,
                       const Bounds2i& pixelBounds,
                       Float rrThreshold = 1,
                       const std::string& lightSampleStrategy = "spatial");

    ///
    /// Main loop. Render the intermediary and final images.
    ///
    void Render(const Scene& scene) override;

    ///
    /// Compute the light arriving along a ray, as usual. It leverages the
    /// incoming radiance cache to sample directions better.
    ///
    Spectrum Li(const RayDifferential& ray, const Scene& scene,
                Sampler &tileSampler, MemoryArena& arena, int depth = 0);

private:
    ///
    /// Create the initial sampling structure.
    ///
    void Preprocess(const Scene& scene);

    ///
    /// Update/reset cache structures.
    ///
    void PostIteration(const Scene& scene, uint iteration,
                       const std::string& id);

    ///
    /// Do one iteration of training + rendering.
    ///
    void RenderIteration(const Scene& scene, uint iteration,
                         const std::string& id);

    ///
    /// Choose a BxDF to sample among all the BxDFs available in bsdf.
    ///
    BxDF* ChooseBxDF(const BSDF &bsdf, Float *pdf, Float u,
                     BxDFType *flags = nullptr);

    ///
    /// Record a path into the recording tree.
    ///
    void RecordPath(const std::vector<PathVertex>& path, Spectrum L);

protected:
    std::shared_ptr<const Camera> camera;

private:
    std::shared_ptr<Sampler> sampler;

    /// Maximum path depth.
    const int maxDepth;

    const Float rrThreshold;
    const std::string lightSampleStrategy;
    std::unique_ptr<LightDistribution> lightDistribution;
    const Bounds2i pixelBounds;

    /// The tree used for sampling directions (reset at each iteration).
    std::shared_ptr<SDTree> samplingTree;

    /// The tree used for recording samples (reset at each iteration).
    std::shared_ptr<SDTree> recordingTree;

    /// The total number of samples available to use.
    const int64_t totalSpp;

    /// What remains from totalSpp.
    int64_t remainingSpp;

    /// The number of samples from each sampling technique.
    struct {
        int64_t bsdf;            ///< Number of samples from the BSDF.
        int64_t radianceField;   ///< Number of samples from 5D tree.
        float   proportion;      ///< Ratio BSDF/(BSDF+Field).
    } nSamples;
};


SDGuidedIntegrator* CreatePPGIntegrator(const ParamSet& params,
    std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}


#endif //PBRT_V3_STORM_SDGUIDED_H
