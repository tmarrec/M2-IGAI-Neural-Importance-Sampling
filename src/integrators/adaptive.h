#ifndef PBRT_INTEGRATORS_ADAPTIVE_H
#define PBRT_INTEGRATORS_ADAPTIVE_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

// integrators/adaptive.h*
#include "pbrt.h"
#include "film.h"
#include "volpath.h"
#include <chrono>
#include <functional>

namespace pbrt {

// _Statistics_ is a reusable class which can be added to an _Integrator_ to
// compute convergence statistics, and control the render loop adaptively.
class Statistics {
    // Three modes are available to render the image:
    // - normal: render using an uniform number of samples throughout the image
    // - error: render with an adaptive number of samples according to the error
    // - time: render with a fixed target computation time
    enum class Mode {NORMAL, ERROR, TIME};

    // A steady clock gives the most robust estimation of computation time
    using Clock = std::chrono::steady_clock;

    // The _SamplingFunctor_ is given by the integrator to the _Statistics_
    // instance. It allows the _Statistics_ instance to draw an arbitrary number
    // of samples at a given pixel. The functor does not take any parameter and
    // returns a Spectrum value, thus its signature is Spectrum(void).
    using SamplingFunctor = std::function<Spectrum()>;

  public:
    // The object keeps a 2D grid of _Pixel_s up to date during the render. A
    // _Pixel_ holds the three base statistics needed to derive others:
    // - number of samples that were drawn at the pixel
    // - mean of the sample distribution
    // - second moment of the sample distribution
    struct Pixel {
        long samples = 0;
        Spectrum mean = 0;
        Spectrum moment2 = 0;
    };

    Statistics(const ParamSet &paramSet, const Film *originalFilm,
               const Sampler &sampler);

    // Utility functions that are called only once during the render
    void RenderBegin();
    void RenderEnd() const;
    void WriteImages() const;

    // Determines if a new batch should be rendered. By default, only one batch
    // will be rendered regardless of the total rendering time and batch index.
    // @param index The index of the batch that is next to be rendered
    // @return If the next batch should be rendered
    bool StartNextBatch(int index);

    // @return The number of samples a batch contains
    long BatchSize() const;

    // This function controls the rendering process using the information given
    // by the _Integrator_ and the statistics. By default, it just draw the
    // same number of samples throughout the image while updating statistics.
    // @param pixel The pixel being rendered, rounded to the nearest integers
    // @param sampleOnce The functor given by the integrator. Calling it
    //                   generates exactly one random sample. Usually a lambda.
    void SamplingLoop(Point2i pixel, const SamplingFunctor &sampleOnce);

    // Update the statistics grid to account for a newly sampled value.
    // @param pixel The image position where the sample was drawn
    // @param L The newly obtained _Spectrum_ value
    void UpdateStats(Point2i pixel, Spectrum &&L);

    // Functions used to setup and update the progress bar
    std::string WorkTitle() const;
    long UpdateWork() const;

  private:
    // The rendering mode as indicated in the .pbrt scene file
    const Mode mode;

    // For all modes, the maximum number of samples that can be drawn
    const long maxSamples;

    // For error mode:
    // - the minimum number of samples used to estimate the statistics
    const long minSamples;
    // - the error threshold that should be reached by each pixel
    const float errorThreshold;
    // - the confidence level (in percent) to use with confidence error heuristic
    const float confidenceLevel;
    // - the zstar value associated with the confidence level
    float zstar;
    // - the chosen error heuristic (relative, standard, confidence...)
    const std::string errorHeuristic;

    // For time mode:
    // - the number of samples a batch contains
    const long batchSize;
    // - the target time
    const long targetSeconds;

    // Original film used to construct the statistics images
    const Film *originalFilm;
    // Film size
    const Bounds2i pixelBounds;
    // Pixel data grid updated with each sample, and exported after the render
    std::unique_ptr<Pixel[]> pixels;

    // Starting time of the render
    Clock::time_point startTime;
    // A boolean value used to render exactly one batch in normal / error modes
    bool batchOnce = true;

    // @param statsPixel The three base statistics
    // @return Either the total number of drawn samples, or the ratio of
    //         additional samples that were used at this position
    Float Sampling(const Pixel &statsPixel) const;

    // @param statsPixel The three base statistics
    // @return The estimated variance
    Spectrum Variance(const Pixel &statsPixel) const;

    // @param statsPixel The three base statistics
    // @return The estimation error according to the chosen heuristic
    Spectrum Error(const Pixel &statsPixel) const;

    // A criterion to determine if a given pixel has converged or not.
    // @return If the pixel should stop being rendered, e.g. if the maximum
    //         number of samples was reached or the error is below the threshold
    bool StopCriterion(Point2i pixel) const;

    // @return The elapsed time from the beginning of the render, in ms
    long ElapsedMilliseconds() const;

    // Access functions to the stored pixels grid
    Pixel &GetPixel(Point2i pixel);
    const Pixel &GetPixel(Point2i pixel) const;

    // A dummy filter to construct the statistics films
    static std::unique_ptr<Filter> StatImagesFilter();
};

// _VolPathAdaptive_ is similar to the original VolPathIntegrator, except for
// the _Render_ function that is configured to use _Statistics_.
class VolPathAdaptive : public VolPathIntegrator {
  public:
    VolPathAdaptive(Statistics stats, int maxDepth,
                    std::shared_ptr<const Camera> camera,
                    std::shared_ptr<Sampler> sampler,
                    const Bounds2i &pixelBounds, Float rrThreshold = 1,
                    const std::string &lightSampleStrategy = "spatial");

    // The only method required by the _Integrator_ interface
    void Render(const Scene &scene) override;

  private:
    // An instance of _Statistics_ that will compute convergence statistics and
    // control the rendering loop according to them
    Statistics stats;
    std::shared_ptr<Sampler> sampler;
    const Bounds2i pixelBounds;
};

VolPathAdaptive *CreateVolPathAdaptive(const ParamSet &params,
                                       std::shared_ptr<Sampler> sampler,
                                       std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_ADAPTIVE_H
