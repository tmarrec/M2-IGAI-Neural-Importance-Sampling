// integrators/adaptive.cpp*
#include "adaptive.h"
#include "camera.h"
#include "filters/box.h"
#include "paramset.h"
#include "progressreporter.h"
#include "stats.h"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
STAT_COUNTER("Integrator/Render time in milliseconds", nElapsedMilliseconds);


/*----------------------------------------------------------------------------*/
/*
 * erf inverse function : taken from https://github.com/lakshayg/erfinv
 */
// Returns a floating point number y such that std::erf(y)
// is close to x. The current implementation is quite accurate
// when x is away from +1.0 and -1.0. As x approaches closer
// to those values, the error in the result increases.
long double erfinv(long double x) {

    if (x < -1 || x > 1) {
        return NAN;
    } else if (x == 1.0) {
        return INFINITY;
    } else if (x == -1.0) {
        return -INFINITY;
    }

    const long double LN2 = 6.931471805599453094172321214581e-1L;

    const long double A0 = 1.1975323115670912564578e0L;
    const long double A1 = 4.7072688112383978012285e1L;
    const long double A2 = 6.9706266534389598238465e2L;
    const long double A3 = 4.8548868893843886794648e3L;
    const long double A4 = 1.6235862515167575384252e4L;
    const long double A5 = 2.3782041382114385731252e4L;
    const long double A6 = 1.1819493347062294404278e4L;
    const long double A7 = 8.8709406962545514830200e2L;

    const long double B0 = 1.0000000000000000000e0L;
    const long double B1 = 4.2313330701600911252e1L;
    const long double B2 = 6.8718700749205790830e2L;
    const long double B3 = 5.3941960214247511077e3L;
    const long double B4 = 2.1213794301586595867e4L;
    const long double B5 = 3.9307895800092710610e4L;
    const long double B6 = 2.8729085735721942674e4L;
    const long double B7 = 5.2264952788528545610e3L;

    const long double C0 = 1.42343711074968357734e0L;
    const long double C1 = 4.63033784615654529590e0L;
    const long double C2 = 5.76949722146069140550e0L;
    const long double C3 = 3.64784832476320460504e0L;
    const long double C4 = 1.27045825245236838258e0L;
    const long double C5 = 2.41780725177450611770e-1L;
    const long double C6 = 2.27238449892691845833e-2L;
    const long double C7 = 7.74545014278341407640e-4L;

    const long double D0 = 1.4142135623730950488016887e0L;
    const long double D1 = 2.9036514445419946173133295e0L;
    const long double D2 = 2.3707661626024532365971225e0L;
    const long double D3 = 9.7547832001787427186894837e-1L;
    const long double D4 = 2.0945065210512749128288442e-1L;
    const long double D5 = 2.1494160384252876777097297e-2L;
    const long double D6 = 7.7441459065157709165577218e-4L;
    const long double D7 = 1.4859850019840355905497876e-9L;

    const long double E0 = 6.65790464350110377720e0L;
    const long double E1 = 5.46378491116411436990e0L;
    const long double E2 = 1.78482653991729133580e0L;
    const long double E3 = 2.96560571828504891230e-1L;
    const long double E4 = 2.65321895265761230930e-2L;
    const long double E5 = 1.24266094738807843860e-3L;
    const long double E6 = 2.71155556874348757815e-5L;
    const long double E7 = 2.01033439929228813265e-7L;

    const long double F0 = 1.414213562373095048801689e0L;
    const long double F1 = 8.482908416595164588112026e-1L;
    const long double F2 = 1.936480946950659106176712e-1L;
    const long double F3 = 2.103693768272068968719679e-2L;
    const long double F4 = 1.112800997078859844711555e-3L;
    const long double F5 = 2.611088405080593625138020e-5L;
    const long double F6 = 2.010321207683943062279931e-7L;
    const long double F7 = 2.891024605872965461538222e-15L;

    long double abs_x = fabsl(x);

    if (abs_x <= 0.85L) {
        long double r = 0.180625L - 0.25L * x * x;
        long double num = (((((((A7 * r + A6) * r + A5) * r + A4) * r + A3) * r + A2) * r + A1) * r + A0);
        long double den = (((((((B7 * r + B6) * r + B5) * r + B4) * r + B3) * r + B2) * r + B1) * r + B0);
        return x * num / den;
    }

    long double r = sqrtl(LN2 - logl(1.0L - abs_x));

    long double num, den;
    if (r <= 5.0L) {
        r = r - 1.6L;
        num = (((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0);
        den = (((((((D7 * r + D6) * r + D5) * r + D4) * r + D3) * r + D2) * r + D1) * r + D0);
    } else {
        r = r - 5.0L;
        num = (((((((E7 * r + E6) * r + E5) * r + E4) * r + E3) * r + E2) * r + E1) * r + E0);
        den = (((((((F7 * r + F6) * r + F5) * r + F4) * r + F3) * r + F2) * r + F1) * r + F0);
    }

    return copysignl(num / den, x);
}

// Refine the result of erfinv by performing Newton-Raphson
// iteration nr_iter number of times. This method works well
// when the value of x is away from 1.0 and -1.0
long double erfinv_refine(long double x, int nr_iter) {
    const long double k = 0.8862269254527580136490837416706L; // 0.5 * sqrt(pi)
    long double y = erfinv(x);
    while (nr_iter-- > 0) {
        y -= k * (erfl(y) - x) / expl(-y * y);
    }
    return y;
}
/*----------------------------------------------------------------------------*/

Statistics::Statistics(const ParamSet &paramSet, const Film *originalFilm,
                       const Sampler &sampler)
    : mode([](std::string &&modeString) {
          return modeString == "time"  ? Mode::TIME
               : modeString == "error" ? Mode::ERROR
               :                         Mode::NORMAL;
      }(paramSet.FindOneString("mode", "normal"))),
      maxSamples(sampler.samplesPerPixel),
      minSamples(paramSet.FindOneInt("minsamples", 64)),
      errorThreshold(paramSet.FindOneFloat("errorthreshold", 0.01)),
      confidenceLevel(paramSet.FindOneFloat("confidencelevel", 95)),
      errorHeuristic(paramSet.FindOneString("errorheuristic", "relative")),
      batchSize(paramSet.FindOneInt("batchsize", 16)),
      targetSeconds(paramSet.FindOneInt("targetseconds", 60)),
      originalFilm(originalFilm),
      pixelBounds(originalFilm->croppedPixelBounds),
      pixels(new Pixel[pixelBounds.Area()]) {
    zstar = std::sqrt(2.) * erfinv_refine( confidenceLevel/100., 100 );
}

void Statistics::RenderBegin() {
    startTime = Clock::now();
}

void Statistics::RenderEnd() const {
    nElapsedMilliseconds = ElapsedMilliseconds();
}

void Statistics::WriteImages() const {
    CHECK(originalFilm);
    Film samplesFilm(*originalFilm, StatImagesFilter(), "_samples");
    Film varianceFilm(*originalFilm, StatImagesFilter(), "_variance");
    Film errorFilm(*originalFilm, StatImagesFilter(), "_error");

    for (Point2i pixel : pixelBounds) {
        Point2f floatPixel(static_cast<Float>(pixel.x),
                           static_cast<Float>(pixel.y));
        const Pixel &statsPixel = GetPixel(pixel);
        samplesFilm.AddSplat(floatPixel, Sampling(statsPixel));
        varianceFilm.AddSplat(floatPixel, Variance(statsPixel));
        errorFilm.AddSplat(floatPixel, Error(statsPixel));
    }

    samplesFilm.WriteImage();
    varianceFilm.WriteImage();
    errorFilm.WriteImage();
}

bool Statistics::StartNextBatch(int index) {
    switch (mode) {
    case Mode::TIME:
        return ElapsedMilliseconds() < targetSeconds * 1000
            && (index + 1) * batchSize < maxSamples;

    default:
        if (batchOnce) {
            batchOnce = false;
            return true;
        } else
            return false;
    }
}

long Statistics::BatchSize() const {
    return batchSize;
}

void Statistics::SamplingLoop(Point2i pixel, const SamplingFunctor &sampleOnce){
    auto loop = [&]() { UpdateStats(pixel, sampleOnce()); };

    switch (mode) {
    case Mode::NORMAL:
        for (long i = 0; i < maxSamples; ++i)
            loop();
        break;

    case Mode::TIME:
        for (long i = 0; i < batchSize; ++i)
            loop();
        break;

    case Mode::ERROR:
        for (long i = 0; i < minSamples; ++i)
            loop();
        while (!StopCriterion(pixel))
            loop();
        break;
    }
}

void Statistics::UpdateStats(Point2i pixel, Spectrum &&L) {
    if (!InsideExclusive(pixel, pixelBounds)) return;

    Pixel &statsPixel = GetPixel(pixel);
    long &samples = statsPixel.samples;
    Spectrum &mean = statsPixel.mean;
    Spectrum &moment2 = statsPixel.moment2;

    samples++;
    Spectrum delta1 = L - mean;
    mean += delta1 / samples;
    Spectrum delta2 = L - mean;
    moment2 += delta1 * delta2;
}

std::string Statistics::WorkTitle() const {
    std::string type = mode == Mode::TIME  ? "time"
                     : mode == Mode::ERROR ? "error"
                     :                       "sampling";
    return "Rendering (equal " + type + ')';
}

long Statistics::UpdateWork() const {
    return mode == Mode::TIME ? 0 : 1;
}

Float Statistics::Sampling(const Pixel &statsPixel) const {
    return mode == Mode::ERROR
         ? static_cast<Float>(statsPixel.samples - minSamples) /
                   static_cast<Float>(maxSamples - minSamples)
         : static_cast<Float>(statsPixel.samples);
}

Spectrum Statistics::Variance(const Pixel &statsPixel) const {
    return statsPixel.samples > 1
         ? statsPixel.moment2 / (static_cast<Float>(statsPixel.samples) - 1)
         : 0;
}

Spectrum Statistics::Error(const Pixel &statsPixel) const {
    auto standardError = statsPixel.samples > 0
                       ? Sqrt(Variance(statsPixel) / statsPixel.samples)
                       : 0;
    if (errorHeuristic == "relative") {
        Float relativeError[3] {
            statsPixel.mean[0] > 1E-6 ? standardError[0] / statsPixel.mean[0] : 0,
            statsPixel.mean[1] > 1E-6 ? standardError[1] / statsPixel.mean[1] : 0,
            statsPixel.mean[2] > 1E-6 ? standardError[2] / statsPixel.mean[2] : 0
        };
        return Spectrum::FromRGB(relativeError);
    }
    if (errorHeuristic == "confidence") {
        return 2.*zstar*standardError;
    }
    return standardError;
}

bool Statistics::StopCriterion(Point2i pixel) const {
    if (!InsideExclusive(pixel, pixelBounds)) return true;
    const Pixel &statsPixel = GetPixel(pixel);
    return statsPixel.samples >= maxSamples
        || Error(statsPixel).y() < errorThreshold;
}

long Statistics::ElapsedMilliseconds() const {
    using namespace std::chrono;
    return duration_cast<milliseconds>(Clock::now() - startTime).count();
}

Statistics::Pixel &Statistics::GetPixel(Point2i pixel) {
    const auto *that = this;
    return const_cast<Pixel &>(that->GetPixel(pixel));
}

const Statistics::Pixel &Statistics::GetPixel(Point2i pixel) const {
    CHECK(InsideExclusive(pixel, pixelBounds));
    int width = pixelBounds.pMax.x - pixelBounds.pMin.x;
    int offset = (pixel.x - pixelBounds.pMin.x)
               + (pixel.y - pixelBounds.pMin.y) * width;
    return pixels[offset];
}

std::unique_ptr<Filter> Statistics::StatImagesFilter() {
    return std::unique_ptr<Filter>(new BoxFilter({0, 0}));
}

VolPathAdaptive::VolPathAdaptive(Statistics stats, int maxDepth,
                                 std::shared_ptr<const Camera> camera,
                                 std::shared_ptr<Sampler> sampler,
                                 const Bounds2i &pixelBounds, Float rrThreshold,
                                 const std::string &lightSampleStrategy)
    : VolPathIntegrator(maxDepth, std::move(camera), sampler, pixelBounds,
                        rrThreshold, lightSampleStrategy),
      stats(std::move(stats)),
      sampler(std::move(sampler)),
      pixelBounds(pixelBounds) {}

void VolPathAdaptive::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);

    stats.RenderBegin();
    ProgressReporter reporter(nTiles.x * nTiles.y, stats.WorkTitle());
    for (int batch = 0; stats.StartNextBatch(batch); ++batch) {
        ParallelFor2D([&](Point2i tile) {
            // Render section of image corresponding to _tile_

            // Allocate _MemoryArena_ for tile
            MemoryArena arena;

            // Get sampler instance for tile
            int seed = nTiles.x * nTiles.y * batch + nTiles.x * tile.y + tile.x;
            std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

            // Compute sample bounds for tile
            int x0 = sampleBounds.pMin.x + tile.x * tileSize;
            int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
            int y0 = sampleBounds.pMin.y + tile.y * tileSize;
            int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
            Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
            LOG(INFO) << "Starting image tile " << tileBounds;

            // Get _FilmTile_ for tile
            auto filmTile = camera->film->GetFilmTile(tileBounds);

            // Loop over pixels in tile to render them
            for (Point2i pixel : tileBounds) {
                {
                    ProfilePhase pp(Prof::StartPixel);
                    tileSampler->StartPixel(pixel);
                    tileSampler->SetSampleNumber(batch * stats.BatchSize());
                }

                // Do this check after the StartPixel() call; this keeps
                // the usage of RNG values from (most) Samplers that use
                // RNGs consistent, which improves reproducibility /
                // debugging.
                if (!InsideExclusive(pixel, pixelBounds))
                    continue;

                stats.SamplingLoop(pixel, [&]() {
                    // Initialize _CameraSample_ for current sample
                    CameraSample cameraSample =
                        tileSampler->GetCameraSample(pixel);

                    // Generate camera ray for current sample
                    RayDifferential ray;
                    Float rayWeight =
                        camera->GenerateRayDifferential(cameraSample, &ray);
                    ray.ScaleDifferentials(
                        1 / std::sqrt((Float)tileSampler->samplesPerPixel));
                    ++nCameraRays;

                    // Evaluate radiance along camera ray
                    Spectrum L(0.f);
                    if (rayWeight > 0)
                        L = Li(ray, scene, *tileSampler, arena, 0);

                    // Add camera ray's contribution to image
                    filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

                    // Free _MemoryArena_ memory from computing image sample
                    // value
                    arena.Reset();

                    tileSampler->StartNextSample();
                    return L;
                });
            }
            LOG(INFO) << "Finished image tile " << tileBounds;

            // Merge image tile into _Film_
            camera->film->MergeFilmTile(std::move(filmTile));
            reporter.Update(stats.UpdateWork());
        }, nTiles);
    }
    reporter.Done();
    LOG(INFO) << "Rendering finished";
    stats.RenderEnd();

    // Save final image after rendering
    camera->film->WriteImage();
    stats.WriteImages();
}

VolPathAdaptive *CreateVolPathAdaptive(const ParamSet &params,
                                       std::shared_ptr<Sampler> sampler,
                                       std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");

    return new VolPathAdaptive({params, camera->film, *sampler}, maxDepth,
                               camera, sampler, pixelBounds, rrThreshold,
                               lightStrategy);
}

}  // namespace pbrt
