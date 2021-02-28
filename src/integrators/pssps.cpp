
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// integrators/pssps.cpp*
#include "integrators/pssps.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "progressreporter.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
STAT_COUNTER("Integrator/Render time in milliseconds", nElapsedMilliseconds);
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PSSPSIntegrator Method Definitions
PSSPSIntegrator::PSSPSIntegrator(int maxDepth,
                               std::shared_ptr<const Camera> camera,
                               std::shared_ptr<Sampler> sampler,
                               const Bounds2i &pixelBounds, Float rrThreshold,
                               const std::string &lightSampleStrategy,
                               int sampleBudget,
                               Strategie strategie)
    : maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampleStrategy(lightSampleStrategy),
      sampler(sampler),
      camera(camera),
      pixelBounds(pixelBounds),
      sampleBudget(sampleBudget),
      strategie(strategie){}

typedef struct stats_s {
	stats_s() : samples(0), mean(0), moment2(0) {}
	stats_s(long samples, Spectrum mean, Spectrum moment2) : samples(samples), mean(mean), moment2(moment2){}
	long samples;
	Spectrum mean;
	Spectrum moment2;
} Stats;

typedef struct {
	Spectrum L;
	Float rayWeight;
	Point2f pFilm;
} Sample;

void UpdateStats(Stats &stats, Spectrum L) {
	++stats.samples;
	Spectrum delta = L - stats.mean;
	stats.mean += delta / stats.samples;
	stats.moment2 += delta * (L - stats.mean);

	// Update the statistics above using Welford's online algorithm
}

Spectrum Variance(const Stats& stat) {
	return stat.moment2 / stat.samples;
}

void PSSPSIntegrator::Render(const Scene &scene) {
	std::cout << "Sample Budget : " << sampleBudget << std::endl;
	std::cout << "Strategie : ";
	switch (strategie) {
		case variance:
			std::cout << "variance" << std::endl;
			break;
		case every_iteration:
			std::cout << "every iteration" << std::endl;
			break;
		case sum_iteration:
			std::cout << "sum iteration" << std::endl;
	}
	lightDistribution =
			CreateLightSampleDistribution(lightSampleStrategy, scene);
	int budget = sampleBudget;
	int samplesPerPixel = 1;
	Float sumWeights = 0;
	// Compute number of tiles, _nTiles_, to use for parallel rendering
	Bounds2i sampleBounds = camera->film->GetSampleBounds();
	Vector2i sampleExtent = sampleBounds.Diagonal();
	const int tileSize = 16;
	Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
	std::vector<std::unique_ptr<FilmTile>> filmTiles;
	std::mutex mutexFilmTiles;
	int filmTilesIdx = 0;

	// stats
	std::vector<Stats> pixelStats(sampleExtent.x * sampleExtent.y);

	std::cout << sampleExtent.x << " " << sampleExtent.y << std::endl;

	while (budget > 0) {
		std::cout << "Buget : " << budget << std::endl;
		std::cout << "Samples per pixel : " << samplesPerPixel << std::endl;

		// generate new paths
		std::cout << "Getting data from neural network." << std::endl;
		std::tie(paths, probas) = nice.get_paths(sampleExtent.x * sampleExtent.y * samplesPerPixel);

		// prepare the buffers
		std::vector<Float> brightnessToNN(probas.size());

		{
			ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
			ParallelFor2D([&](Point2i tile) {
                // Render section of image corresponding to _tile_

                // Allocate _MemoryArena_ for tile
                MemoryArena arena;

				std::unique_ptr<FilmTile> filmTile = camera->film->GetFilmTile(sampleBounds);

				// Get sampler instance for tile
				int seed = tile.y * nTiles.x + tile.x;
				std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

				// Compute sample bounds for tile
				int x0 = sampleBounds.pMin.x + tile.x * tileSize;
				int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
				int y0 = sampleBounds.pMin.y + tile.y * tileSize;
				int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
				Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));

				LOG(INFO) << "Starting image tile " << tileBounds;


				// Offset in sampleBuffer
				int pixelIdx = tile.y * tileSize * sampleExtent.x + tile.x * (y1 - y0) * tileSize;

				// Loop over pixels in tile to render them
				for (Point2i pixel : tileBounds) {
					{
						ProfilePhase pp(Prof::StartPixel);
						tileSampler->StartPixel(pixel);
						tileSampler->samplesPerPixel = samplesPerPixel;
					}


                    // Do this check after the StartPixel() call; this keeps
                    // the usage of RNG values from (most) Samplers that use
                    // RNGs consistent, which improves reproducability /
                    // debugging.
                    if (!InsideExclusive(pixel, pixelBounds))
                        continue;

                    // Offset in paths and probas
                    int sampleIdx = pixelIdx * samplesPerPixel;
                    Stats pixelStat;  // = pixelStats[pixelIdx];

					do {

						// Initialize _CameraSample_ for current sample
						CameraSample cameraSample =
								tileSampler->GetCameraSample(pixel);

						// Generate camera ray for current sample
                        RayDifferential ray;
                        Float rayWeight =
                            camera->GenerateRayDifferential(cameraSample, &ray);
                        ray.ScaleDifferentials(
                            1 / std::sqrt(samplesPerPixel));
                        ++nCameraRays;

                        // Evaluate radiance along camera ray
                        Spectrum L(0.f);
                        // Li's last parameter is supposed to be the depth but is never used in this case, so it's just getting replaced by the pixel index for the neural sampling
                        if (rayWeight > 0) L = Li(ray, scene, *tileSampler, arena, sampleIdx);

                        // Issue warning if unexpected radiance value returned
                        if (L.HasNaNs()) {
                            LOG(ERROR) << StringPrintf(
                                "Not-a-number radiance value returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L = Spectrum(0.f);
                        } else if (L.y() < -1e-5) {
                            LOG(ERROR) << StringPrintf(
                                "Negative luminance value, %f, returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                L.y(), pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L = Spectrum(0.f);
                        } else if (std::isinf(L.y())) {
                            LOG(ERROR) << StringPrintf(
                                "Infinite luminance value returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                            L = Spectrum(0.f);
                        }
                        VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " <<
                            ray << " -> L = " << L;

						// Free _MemoryArena_ memory from computing image sample
						// value
						arena.Reset();

						// Add camera ray's contribution to image
						filmTile->AddSample(cameraSample.pFilm, L, rayWeight);
						L *= rayWeight;
						brightnessToNN[sampleIdx] = L.y();
						if (strategie == variance) {
							UpdateStats(pixelStat, L);
						}
						++sampleIdx;
					} while (tileSampler->StartNextSample());
					pixelStats[pixelIdx] = pixelStat;
					++pixelIdx;
                }
                LOG(INFO) << "Finished image tile " << tileBounds;
				if (strategie == every_iteration or strategie == sum_iteration) {
					camera->film->MergeFilmTile(std::move(filmTile));
				}
				if (strategie == variance) {
					std::lock_guard<std::mutex> guard(mutexFilmTiles);
					filmTiles.emplace_back(std::move(filmTile));
				}
                reporter.Update();
			}, nTiles);
			reporter.Done();
        }
		if (strategie == variance) {
			Float weight;
			if (samplesPerPixel > 1) {
				Spectrum meanVar(0.0f);
				for (int i = 0; i < pixelStats.size(); ++i) {
					meanVar += Variance(pixelStats[i]);
				}
				meanVar /= pixelStats.size();
				Float var = meanVar.y();
				weight = 1 / var;
			} else {
				weight = 1.f;
			}
			sumWeights += samplesPerPixel * weight;
			for (;filmTilesIdx < filmTiles.size(); ++filmTilesIdx) {
				filmTiles[filmTilesIdx]->ScaleBy(weight);
			}
		}
		if (strategie == every_iteration) {
			camera->film->WriteImage(1, std::to_string(samplesPerPixel));
		}
		budget -= samplesPerPixel;
    	samplesPerPixel *= 2;

		// Sending brightness to neural network
		if (budget > 0) {
			std::cout << "Sending data to neural network." << std::endl;
			nice.learn(paths, brightnessToNN);
		}
	}
	if (strategie == variance) {
		for (auto &filmTile : filmTiles) {
			filmTile->ScaleBy(sampleBudget / sumWeights);
			camera->film->MergeFilmTile(std::move(filmTile));
		}
	}
	LOG(INFO) << "Rendering finished";
	// Save final image after rendering
	if (strategie == variance or strategie == sum_iteration) {
		camera->film->WriteImage();
	}
}

Spectrum PSSPSIntegrator::Li(const RayDifferential &r, const Scene &scene,
                            Sampler &sampler, MemoryArena &arena,
                            int sampleOffset) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;
	beta /= probas[sampleOffset];
    for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
                << ", beta = " << beta;

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * isect.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            } else {
                for (const auto &light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection || bounces >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        const Distribution1D *distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
            0) {
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
                                                       sampler, false, distrib);
            VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d, wi;
        Float pdf;
        BxDFType flags;
        // Get path direction from the matrix (NICE NN)
		Point2f u;
		if (bounces < nice.path_length) {
			u = {paths[sampleOffset][2 * bounces], paths[sampleOffset][2 * bounces + 1]};
		} else {
        	u = sampler.Get2D();
        }
		Spectrum f = isect.bsdf->Sample_f(wo, &wi, u, &pdf,
										  BSDF_ALL, &flags);
        VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        // pdf is computed at the end 
        //beta *= f * AbsDot(wi, isect.shading.n) / pdf;
        beta *= f * AbsDot(wi, isect.shading.n);
        VLOG(2) << "Updated beta = " << beta;
        CHECK_GE(beta.y(), 0.f);
        DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }
        ray = isect.SpawnRay(wi);

        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            // pdf is computed at the end 
            //beta *= S / pdf;
            beta *= S;

            // Account for the direct subsurface scattering component
            L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                                              lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component
            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                                           BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0) break;
            // pdf is computed at the end 
            //beta *= f * AbsDot(wi, pi.shading.n) / pdf;
            beta *= f * AbsDot(wi, pi.shading.n);
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }
        // Path probability is added as a weight
        // beta *= probas[sampleOffset];

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }
    ReportValue(pathLength, bounces);
    return L;
}

PSSPSIntegrator *CreatePSSPSIntegrator(const ParamSet &params,
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
    int sampleBudget = params.FindOneInt("samplebudget", 4);
    std::cout << params.FindOneInt("xresolution", 4) << std::endl;

    std::string s_strategie = params.FindOneString("strategie", "variance");
    Strategie strategie;
	if (s_strategie == "every_iteration") {
    	strategie = every_iteration;
    } else if (s_strategie == "sum_iteration") {
    	strategie = sum_iteration;
    } else {
    	strategie = variance;
    }

    return new PSSPSIntegrator(maxDepth, camera, sampler, pixelBounds,
                              rrThreshold, lightStrategy, sampleBudget, strategie);
}

}  // namespace pbrt
