
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_PSSPG_H
#define PBRT_INTEGRATORS_PSSPG_H

// integrators/psspg.h*
#include "pbrt.h"
#include "integrator.h" 
#include "lightdistrib.h"

namespace pbrt {

// PSSGPTIntegrator declerations
class PSSGPTIntegrator : public Integrator {
public:
	// PSSGPTIntegrator public methods
	PSSGPTIntegrator(std::shared_ptr<const Camera> _camera, std::shared_ptr<Sampler> _s,
				   int _maxDepth, int _rrD, const Bounds2i &_pixelBounds, int _kdMaxInLeaf,
				   int _totalIter, int _finalIterSPP, int _kdMD, Float _rrT = 1.f,
				   bool _NEE = false, int _lgDim = 2, int _initSpl = 64,
				   Float _kdNodeVal = 10.f, Float _guideP = 0.5f,
				   const std::string &_lightSampleStrategy = "spatial")
		: sampler(_s), camera(_camera), maxDepth(_maxDepth), kdMaxDepth(_kdMD),
		  pixelBounds(_pixelBounds), kdMaxInLeaf(_kdMaxInLeaf), rrThreshold(_rrT), 
		  rrDepth(_rrD), kdNodeVal(_kdNodeVal), totalIter(_totalIter), lgDim(_lgDim),
		  finalSPP(_finalIterSPP), initSpl(_initSpl), NEE(_NEE), guideProb(_guideP),
		  lightSampleStrategy(_lightSampleStrategy) {}
	~PSSGPTIntegrator() {}

	void Preprocess(const Scene &_scene, Sampler &_sampler);
	void Render(const Scene &_scene);
	void CheckRadiance(Spectrum &L, const Point2i &pixel, int cSample);
	Spectrum Li(const RayDifferential &_ray, const Scene &_scene,
		        Sampler &_sampler, MemoryArena &_arena,
		        const std::vector<Point2f> &samples, bool misInfo,
				const std::vector<Float>& rndPdf, Spectrum &LnP);
	
private:
	// PSSGPTIntegrator private data
	const bool NEE;
	const Bounds2i pixelBounds;
	std::shared_ptr<Sampler> sampler;
	std::shared_ptr<const Camera> camera;
	const std::string lightSampleStrategy;
	const Float rrThreshold, kdNodeVal, guideProb;
	const unsigned lgDim, kdMaxInLeaf, kdMaxDepth;
	std::unique_ptr<LightDistribution> lightDistribution;
	const int maxDepth, rrDepth, totalIter, initSpl, finalSPP;
};

PSSGPTIntegrator *CreatePSSGPTIntegrator(
					const ParamSet &params,
					std::shared_ptr<Sampler> sampler,
					std::shared_ptr<const Camera> camera);

}	// namespace pbrt

#endif // PBRT_INTEGRATORS_PSSPG_H