
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLERS_NEURAL_H
#define PBRT_SAMPLERS_NEURAL_H

// samplers/neural.h*
#include "sampler.h"
#include "rng.h"
#include "NICE.h"

namespace pbrt {

class NeuralSampler : public Sampler {
  public:
    NeuralSampler(int ns, int seed = 0);
    void StartPixel(const Point2i &);
    Float Get1D();
    Point2f Get2D();
    Point2f GetNeural2D();
    Point2f GetUniform2D();
    std::unique_ptr<Sampler> Clone(int seed);
    void learn(std::vector<std::vector<float>> chemins, std::vector<float> luminance);
    std::tuple<std::vector<std::vector<float>>, std::vector<float>> get_paths(unsigned int num_path);
    
    std::shared_ptr<NICE> nice;
  private:
    RNG rng;
};

Sampler *CreateNeuralSampler(const ParamSet &params);

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_NEURAL_H
