
/*
	pbrt source code is Copyright(c) 1998-2016
						Matt Pharr, Greg Humphreys, and Wenzel Jakob.

	This file is part of pbrt.
	
	This file is a re-implementation of the paper
	"Primary Sample Space Path Guiding" DOI = {10.2312/sre.20181174}.
	This implementation is provided by Jerry Guo, TU Delft and is
	Copyright(c) 2018 CGV, TU Delft

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

// integrators/psspg.cpp*
#include "integrators/PSSPG-Guoetal-2018/psspg.h"
#include "integrators/PSSPG-Guoetal-2018/hdkdtree.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "filters/box.h"
#include "samplers/random.h"
#include "paramset.h"
#include "progressreporter.h"
#include "parallel.h"
#include "scene.h"
#include "stats.h"


#define DIM 4
//#define DIM 2

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

void PSSGPTIntegrator::Preprocess(const Scene & scene, Sampler & sampler) {
	lightDistribution =
		CreateLightSampleDistribution(lightSampleStrategy, scene);
}

void PSSGPTIntegrator::Render(const Scene & scene) {
	Preprocess(scene, *sampler);
	// Render image tiles in parallel

	// Compute number of tiles, _nTiles_, to use for parallel rendering
	Bounds2i sampleBounds = camera->film->GetSampleBounds();
	Vector2i sampleExtent = sampleBounds.Diagonal();
	const int tileSize = 16;
	Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
				   (sampleExtent.y + tileSize - 1) / tileSize);
	// Make a random sampler for HDKdTree
	std::shared_ptr<Sampler> HDKdSampler = nullptr;
	HDKdSampler = std::shared_ptr<Sampler>(new RandomSampler(10240));
	ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
	{
		ParallelFor2D([&](Point2i tile) {
			// Render section of image corresponding to _tile_
			// Allocate _MemoryArena_ for tile
			MemoryArena arena;

			// Get sampler instance for tile
			int seed = tile.y * nTiles.x + tile.x;
			std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);
			std::unique_ptr<Sampler> kdSampler = HDKdSampler->Clone(seed);

			// Compute sample bounds for tile
			int x0 = sampleBounds.pMin.x + tile.x * tileSize;
			int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
			int y0 = sampleBounds.pMin.y + tile.y * tileSize;
			int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
			Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
			LOG(INFO) << "Starting image tile " << tileBounds;

			// Get _FilmTile_ for tile
			std::unique_ptr<FilmTile> filmTile =
				camera->film->GetFilmTile(tileBounds);

			// Make a HDKdTree for current tile
			HDKdTree<DIM> kdTree(kdNodeVal, kdMaxDepth);
			// Make a vector variable to collect samples
			std::vector<Point2f> samples(DIM / 2);
			std::vector<Float> pSample(DIM, 1.f);
			int cSPP = initSpl;
			
			for (int iter = 0; iter < totalIter; iter++) {
				// Double cSPP in training iterations
				// or use finalSPP for rendering iteration
				cSPP = (iter == (totalIter - 1)) ? finalSPP : cSPP;
				for (Point2i pixel : tileBounds) {
					{
						ProfilePhase pp(Prof::StartPixel);
						tileSampler->StartPixel(pixel);
						kdSampler->StartPixel(pixel);
						const int iterSpl0 = initSpl * (std::pow(2, iter) - 1);
						tileSampler->SetSampleNumber(iterSpl0);
						kdSampler->SetSampleNumber(iterSpl0);
					}
					
					for (int n = 0; n < cSPP; n++) {
						// Initialize _CameraSample_ for current sample
						CameraSample camSample = 
							tileSampler->GetCameraSample(pixel);
						// Generate camera ray for current sample
						RayDifferential ray;
						Float rayWeight =
							camera->GenerateRayDifferential(camSample, &ray);
						ray.ScaleDifferentials(
							1 / std::sqrt((Float)tileSampler->samplesPerPixel));
						++nCameraRays;

						// Fill in the _samples_ vector with canonic samples
						// _pSample_ is reset to {1.0f}^D in kdTree.SampleTree()
						for (size_t i = 0; i < DIM / 2; ++i)
							samples[i] = tileSampler->Get2D();

						// ==========================================================
						//				Multiple importance sampling
						// ==========================================================
						// Combination of canonical samples and samples from kdTree
						bool misInfo = true;
						if (tileSampler->Get1D() < guideProb) {
							// Use transformed samples from KdTree
							kdTree.SampleTree(*kdSampler, samples, pSample);
						} else {
							// Use the original samples, need _pSample_
							kdTree.SampleTree(samples, pSample);
							misInfo = false;
						}

						// Evaluate radiance along camera ray
						Spectrum L(0.f);			// L * BSDF * G / pdf
						Spectrum LnP(0.f);			// L * BSDF * G
						if (rayWeight > 0)
							L = Li(ray, scene, *tileSampler, arena, samples,
								misInfo, pSample, LnP);

						// Issue warning if unexpected radiance value returned
						CheckRadiance(L,   pixel, tileSampler->CurrentSampleNumber());
						CheckRadiance(LnP, pixel, tileSampler->CurrentSampleNumber());
						VLOG(1) << "Camera sample: " << camSample << " -> ray: " <<
							ray << " -> L = " << L;

						// Make a PathRecord variable to collect data
						// Possibly collect PathRecord in the HDKdTree and iterate
						if (iter < totalIter - 1) {
							PathRecord<DIM> pRecord(samples, LnP);
							kdTree.AddPathRecord(pRecord);
						} else {
							// Add camera ray's contribution to image
							// Only use the last iteration
							filmTile->AddSample(camSample.pFilm, L, rayWeight);
						}
						// Free _MemoryArena_ memory from computing image sample value
						arena.Reset();
						kdSampler->StartNextSample();
						tileSampler->StartNextSample();
					}

				}
				cSPP <<= 1;
				kdTree.Iterate();
			}
			LOG(INFO) << "Finished image tile " << tileBounds;

			// Merge image tile into _Film_
			camera->film->MergeFilmTile(std::move(filmTile));
			reporter.Update();
		}, Point2i(nTiles.x, nTiles.y));
		reporter.Done();
	}
	LOG(INFO) << "Rendering finished";

	// Save final image after rendering
	camera->film->WriteImage();
}

void PSSGPTIntegrator::CheckRadiance(Spectrum &L, const Point2i &p, int cs) {
	if (L.HasNaNs()) {
		LOG(ERROR) << StringPrintf(
			"Not-a-number radiance value returned "
			"for pixel (%d, %d), sample %d. Setting to black.",
			p.x, p.y, cs);
		L = Spectrum(0.f);
	} else if (L.y() < -1e-5) {
		LOG(ERROR) << StringPrintf(
			"Negative luminance value, %f, returned "
			"for pixel (%d, %d), sample %d. Setting to black.",
			L.y(), p.x, p.y, cs);
		L = Spectrum(0.f);
	} else if (std::isinf(L.y())) {
		LOG(ERROR) << StringPrintf(
			"Infinite luminance value returned "
			"for pixel (%d, %d), sample %d. Setting to black.",
			p.x, p.y, cs);
		L = Spectrum(0.f);
	}
}

Spectrum PSSGPTIntegrator::Li(const RayDifferential & r, const Scene &scene,
							Sampler & sampler, MemoryArena & arena,
							const std::vector<Point2f> &samples, bool misInfo,
							const std::vector<Float> &rndPdf, Spectrum& LnP) {
	ProfilePhase p(Prof::SamplerIntegratorLi);
	Spectrum L(0.f), beta(1.f), beta1(1.f), beta1NoProb(1.f), E0(0.f);
	RayDifferential ray(r);
	bool specularBounce = false;
	int bounces;
	Float etaScale = 1;
	auto mis = [](float p0, float p1, float ps) { return (p0 + p1) * ps; };

	for (bounces = 0;; ++bounces) {
		// Find next path vertex and accumulate contribution
		VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
				<< ", beta = " << beta;

		// Intersect _ray_ with scene and store intersection in _isect_
		SurfaceInteraction isect;
		bool foundIntersection = scene.Intersect(ray, &isect);
		const bool weCare = bounces < int(DIM / 2);

		// ===================================================================
		//							  Add Emission
		// ===================================================================
		bool addEmmision = !NEE || (NEE && (bounces == 0 || specularBounce));
		// Possibly add emitted light at intersection
		if (addEmmision) {
			// Add emitted light at path vertex or from the environment
			if (foundIntersection) {
				L += beta * isect.Le(-ray.d);
				E0 += weCare ? isect.Le(-ray.d) : Spectrum(0.f);
				VLOG(2) << "Added Le -> L = " << L;
			} else {
				for (const auto &light : scene.infiniteLights) {
					L += beta * light->Le(ray);
				}
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

		// ===================================================================
		//					   Direct illumination sampling
		// ===================================================================
		const Distribution1D *distrib = lightDistribution->Lookup(isect.p);
		// After guided bounces, NEE is turned on by default
		const bool dSampling = NEE ? true : weCare ?
									false : isect.bsdf->NumComponents(
									BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0;
		// Sample illumination from lights to find path contribution.
		// (But skip this for perfectly specular BSDFs.)
		if (dSampling) {
			++totalPaths;
			Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
				sampler, false, distrib);
			VLOG(2) << "Sampled direct lighting Ld = " << Ld;
			if (Ld.IsBlack()) ++zeroRadiancePaths;
			CHECK_GE(Ld.y(), 0.f);
			L += Ld;
		}

		// ===================================================================
		//						      BSDF sampling
		// ===================================================================
		// Sample BSDF to get new path direction
		Vector3f wo = -ray.d, wi;
		Float pdf;
		BxDFType flags;
		
		// Prepare samples for current scattering event
		Point2f currentSample(0.f, 0.f);
		Float pdf_sample = 1.f;
		// Use pregenerated samples when available or use sampler.Get2D()
		if (bounces < (int)samples.size()) {
			currentSample = samples[bounces];
			// MIS combined pdf
			pdf_sample = rndPdf[bounces * 2] * rndPdf[bounces * 2 + 1];
			pdf_sample = mis(pdf_sample, 1.f, guideProb);
			VLOG(2) << "Using samples from HDKdTree, currentSample = "
					<< currentSample << ", pdf_sample = " << pdf_sample;
		} else {
			currentSample = sampler.Get2D();
		}
		
		// Update path throughput
		Spectrum f = isect.bsdf->Sample_f(wo, &wi, currentSample, &pdf,
										  BSDF_ALL, &flags);
		VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
		if (f.IsBlack() || pdf == 0.f || pdf_sample == 0.f) break;
		beta *= f * AbsDot(wi, isect.shading.n) / (pdf * pdf_sample);
		if (weCare) {
			beta1 *= f * AbsDot(wi, isect.shading.n) / (pdf * pdf_sample);
			beta1NoProb *= f * AbsDot(wi, isect.shading.n);
		}
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

		// Account for subsurface scattering, if applicable <- no
		if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
			// Importance sample the BSSRDF
			SurfaceInteraction pi;
			Spectrum S = isect.bssrdf->Sample_S(
				scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
			DCHECK(!std::isinf(beta.y()));
			if (S.IsBlack() || pdf == 0) break;
			beta *= S / pdf;

			// Account for the direct subsurface scattering component
			L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
											  lightDistribution->Lookup(pi.p));

			// Account for the indirect subsurface scattering component
			Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
				BSDF_ALL, &flags);
			if (f.IsBlack() || pdf == 0) break;
			beta *= f * AbsDot(wi, pi.shading.n) / pdf;
			DCHECK(!std::isinf(beta.y()));
			specularBounce = (flags & BSDF_SPECULAR) != 0;
			ray = pi.SpawnRay(wi);
		}

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
	// Factor out stuff and get L1 (some ugly code due to Spectrum division)
	if (!L.IsBlack()) {
		const Spectrum L1 = beta1NoProb * (L - E0);
		LnP[0] = beta1[0] != 0.f ? L1[0] / beta1[0] : 0.f;
		LnP[1] = beta1[1] != 0.f ? L1[1] / beta1[1] : 0.f;
		LnP[2] = beta1[2] != 0.f ? L1[2] / beta1[2] : 0.f;
	}
		
	return L;
}

PSSGPTIntegrator * CreatePSSGPTIntegrator(
					const ParamSet & params,
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
				Bounds2i{ { pb[0], pb[2] },{ pb[1], pb[3] } });
			if (pixelBounds.Area() == 0)
				Error("Degenerate \"pixelbounds\" specified.");
		}
	}
	
	int lgDim		  = params.FindOneInt("LgDim", 4);		// Guided dims
	int rrDepth		  = params.FindOneInt("rrDepth", 3);
	int kdMaxInLeaf   = params.FindOneInt("KdMaxInLeaf", 8);
	int kdMaxDepth    = params.FindOneInt("KdMaxDepth", 12);
	int totalIters    = params.FindOneInt("TotalIter", 4);
	int finalIterSPP  = params.FindOneInt("FinalIterSample", 256);
	int initSample    = params.FindOneInt("InitSample", 64);
	bool NEE	  = params.FindOneBool("NEE", false);
	Float guideProb	  = params.FindOneFloat("GuideProb", 0.5f);
	Float rrThreshold = params.FindOneFloat("rrthreshold", 1.f);
	Float kdNodeValue = params.FindOneFloat("KdNodeValue", 10.f);

	std::string lightStrategy = params.FindOneString("lightsamplestrategy",
													 "spatial");

	return new PSSGPTIntegrator( camera, sampler, maxDepth, rrDepth, pixelBounds,
							   kdMaxInLeaf, totalIters, finalIterSPP, kdMaxDepth, 
							   rrThreshold, NEE, lgDim, initSample, kdNodeValue, 
							   guideProb, lightStrategy );
}
}	// namespace pbrt
