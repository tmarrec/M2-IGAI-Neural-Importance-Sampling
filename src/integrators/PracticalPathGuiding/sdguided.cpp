//
// Created by hrens on 5/4/18.
//

#include "sdguided.h"

#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

#include "filters/box.h"
#include "sampling.h"
#include "parallel.h"
#include "film.h"
#include "sampler.h"
#include "integrator.h"
#include "progressreporter.h"

#include <algorithm>


namespace pbrt {

STAT_COUNTER("Integrator/Iterations", nIterations);
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_INT_DISTRIBUTION("Integrator/Resample", resampleRequired);

using namespace std;


SDGuidedIntegrator::SDGuidedIntegrator(int maxDepth,
                                       shared_ptr<const Camera> camera,
                                       shared_ptr<Sampler> sampler,
                                       const Bounds2i &pixelBounds,
                                       Float rrThreshold,
                                       const string &lightSampleStrategy)
        : camera(camera), sampler(sampler), maxDepth(maxDepth), rrThreshold(rrThreshold),
          lightSampleStrategy(lightSampleStrategy), pixelBounds(pixelBounds), totalSpp(sampler->samplesPerPixel),
          remainingSpp(totalSpp) {
    // Nothing.
}


void SDGuidedIntegrator::Render(const pbrt::Scene &scene) {

    const uint passes = std::ceil(std::log2(sampler->samplesPerPixel));
    LOG(WARNING) << "Rendering with " << sampler->samplesPerPixel << " spp and"
               << " thus run " << passes << " passes" << endl;


    // Create the light distribution, and initialise the 5D data structures.
    Preprocess(scene);

    for (uint pass = 0; pass < passes; ++pass) {
        // Establish number of samples from each technique.
        //nSamples.bsdf = static_cast<int64_t>(floor(nSamples.proportion*(float)sampler->samplesPerPixel));
        //nSamples.radianceField = sampler->samplesPerPixel - nSamples.bsdf;

        auto id = to_string(uint(std::pow(2,pass)) )+"spp-";

        RenderIteration(scene, pass, id);
        PostIteration(scene, pass, id);
    }
#if 0
    // Show the maximum tree.
    const auto& biggestTree = std::max_element(
            samplingTree->tree.quadtrees.begin(), samplingTree->tree.quadtrees.end(),
                                  [](const SDTree::DirectionalTree& A,
                                     const SDTree::DirectionalTree& B) -> bool {
                                        return A.nodes.size() < B.nodes.size();
                                  });
    biggestTree->PrintBoxes();
#endif
}


void SDGuidedIntegrator::Preprocess(const pbrt::Scene &scene) {
    lightDistribution =
            CreateLightSampleDistribution(lightSampleStrategy, scene);

    // TODO See the general problematic of modifying spp on the fly...
    // This sort of works for now.
    *const_cast<int64_t*>(&sampler->samplesPerPixel) = 1;

    // Create the spatial/directional trees.
    samplingTree.reset(new SDTree(scene.WorldBound()));
    recordingTree.reset(new SDTree(scene.WorldBound()));

    // Define the ratio of samples from BSDF and from Light field.
    nSamples.proportion = .5f;
}


void SDGuidedIntegrator::PostIteration(const Scene &scene, uint iteration,
                                       const std::string &id) {
    remainingSpp -= sampler->samplesPerPixel;
#ifdef TREE_DEBUG
    samplingTree->ToObj(id+"struct");
#endif

    // Refine the incident radiance approximation.
    // Notice that this implementation is quite naive in comparison with the
    // authors' version. They use the same spatial distribution for sampling and
    // recording, and the spatial tree contains two lists of quadtrees.
    samplingTree = recordingTree;
    recordingTree.reset(new SDTree(scene.WorldBound()));
    recordingTree->Refine(*samplingTree, iteration);

    // Control the number of samples.
    // TODO Find a reliable way of doing this. Just modifying the spp might
    // totally lead to an undefined behavior, if for instance the sampler
    // decides the value of a parameter at construction according to this
    // value.
    *const_cast<int64_t*>(&sampler->samplesPerPixel) = sampler->samplesPerPixel * 2;

    // Write the film to an intermediary image.
    camera->film->WriteImage(1, id);

    // Clear the film.
    camera->film->Clear();
}


void SDGuidedIntegrator::RenderIteration(const pbrt::Scene &scene,
                                         uint iteration, const std::string &id){
    LOG(WARNING) << "Rendering with " << sampler->samplesPerPixel << " spp" << endl;

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
    {
        ParallelFor2D([&](Point2i tile) {
            // Render section of image corresponding to _tile_

            // Allocate _MemoryArena_ for tile
            MemoryArena arena;

            // Get sampler instance for tile
            int seed = tile.y * nTiles.x + tile.x;
            unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

            // Compute sample bounds for tile
            int x0 = sampleBounds.pMin.x + tile.x * tileSize;
            int x1 = min(x0 + tileSize, sampleBounds.pMax.x);
            int y0 = sampleBounds.pMin.y + tile.y * tileSize;
            int y1 = min(y0 + tileSize, sampleBounds.pMax.y);
            Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));

            // Get _FilmTile_ for tile
            unique_ptr<FilmTile> filmTile =
                    camera->film->GetFilmTile(tileBounds);

            // Loop over pixels in tile to render them
            for (Point2i pixel : tileBounds) {
                {
                    ProfilePhase pp(Prof::StartPixel);
                    tileSampler->StartPixel(pixel);
                }

                // Do this check after the BeginPixel() call; this keeps
                // the usage of RNG values from (most) Samplers that use
                // RNGs consistent, which improves reproducability /
                // debugging.
                if (!InsideExclusive(pixel, pixelBounds)) {
                    continue;
                }

                do {

                    // Initialize _CameraSample_ for current sample
                    CameraSample cameraSample =
                            tileSampler->GetCameraSample(pixel);

                    // Generate camera ray for current sample
                    RayDifferential ray;
                    Float rayWeight =
                            camera->GenerateRayDifferential(cameraSample, &ray);
                    ray.ScaleDifferentials(
                            1 / sqrt((Float)tileSampler->samplesPerPixel));
                    ++nCameraRays;

                    // begin sample
                    // Evaluate radiance along camera ray
                    Spectrum L(0.f);

                    if (rayWeight > 0) {
//                        LOG(WARNING) << "New light estimate" << endl;
                        L = Li(ray, scene, *tileSampler, arena);
                    }

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
                    } else if (isinf(L.y())) {
                        LOG(ERROR) << StringPrintf(
                                "Infinite luminance value returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                pixel.x, pixel.y,
                                (int)tileSampler->CurrentSampleNumber());
                        L = Spectrum(0.f);
                    }
                    VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " <<
                            ray << " -> L = " << L;

                    // Add camera ray's contribution to image
                    filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

                    // Free _MemoryArena_ memory from computing image sample
                    // value
                    arena.Reset();
                } while (tileSampler->StartNextSample());
            }

            LOG(INFO) << "Finished image tile " << tileBounds;

            // Merge image tile into _Film_
            camera->film->MergeFilmTile(move(filmTile));

            reporter.Update();
        }, nTiles);
        reporter.Done();
    }
    LOG(INFO) << "Rendering finished";
    //extractor->Flush();
}

/*
BxDF* SDGuidedIntegrator::ChooseBxDF(const BSDF &bsdf, Float *pdf, Float u,
                                     BxDFType *flags)
{
    int matchingComps = bsdf.NumComponents(BSDF_ALL);
    if (matchingComps == 0) {
        *pdf = 0;
        if (flags)
            *flags = BxDFType(0);
        return nullptr;
    }

    int comp = std::min(static_cast<int>(std::floor(u * matchingComps)),
                        matchingComps - 1);

    // Get _BxDF_ pointer for chosen component
    int count = comp;
    for (int i = 0; i < bsdf.nBxDFs; ++i) {
        if (bsdf.bxdfs[i]->MatchesFlags(BSDF_ALL) && count-- == 0) {
            return bsdf.bxdfs[i];
        }
    }

    return nullptr;
}
*/

Spectrum SDGuidedIntegrator::Li(const pbrt::RayDifferential &r,
                                const pbrt::Scene &scene,
                                Sampler &tileSampler,
                                pbrt::MemoryArena &arena,
                                int depth) {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    vector<PathVertex> path;

    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;

    Float etaScale = 1;

    for (bounces = 0;; ++ bounces) {
        // Find the next intersection.
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        Spectrum vertexEmit(0.f);

        // If this is the first bounce (e.g we see a light directly) or if it
        // is a specular bounce (meaning that we are not able to sample direct
        // lighting) add the light that is emitted by the emissive body or an
        // infinite light.
        if (bounces == 0 || specularBounce) {
            if (foundIntersection) {
                vertexEmit += beta * isect.Le(-ray.d);
            } else {
                for (const auto &infiniteLight : scene.infiniteLights) {
                    vertexEmit += beta *= infiniteLight->Le(ray);
                }
            }
        }

        L += vertexEmit;

        // Terminate if needed.
        if (! foundIntersection || bounces >= maxDepth)
            break;

        // Compute scattering functions and skip over medium boundaries.
        isect.ComputeScatteringFunctions(ray, arena, true);

        // Exit if the BSDF is null, possibly meaning it is a medium boundary.
        if (! isect.bsdf) {
            ray = isect.SpawnRay(ray.d);
            -- bounces;
            continue;
        }

        const Distribution1D *distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR))>0) {
            ++totalPaths;
            Spectrum Ld = UniformSampleOneLight(isect, scene, arena,
                                                tileSampler, false, distrib);

            // Record the new path that was generated WITHOUT taking the beta
            // attenuation (it will be handled by the record function).
            RecordPath(path, Ld);

            // Apply the beta now.
            Ld *= beta;

            if (Ld.IsBlack())
                ++zeroRadiancePaths;

            CHECK_GE(Ld.y(), 0.f);
            L += Ld;

        }

        // Sample BSDF to get new path direction
        Float pdf, pdfL = 0, pdfBsdf = 0, misW = 1.f;
        Vector3f wo = -ray.d, wi;
        BxDFType flags = BSDF_ALL;

        Spectrum f;
        Point2f u = tileSampler.Get2D();

        // Lambda computing MIS weights with the (2-)power heuristic.
        auto misPower = [](Float pa, Float pb) -> Float {
            pa *= pa;
            pb *= pb;
            return pa / (pa + pb);
        };

        /*
        // Same but using some defined proportion (na/na+nb = alpha).
        auto misPowerAlpha = [](Float pa, Float pb, Float alpha=.5f) -> Float {
            pa *= pa;
            pb *= pb;
            return pa / (pa + pb*((1/static_cast<Float>(alpha))-1));
        };
        */

        // Sample the radiance field.
        if (u.x > nSamples.proportion) {
            u.x = (u.x - nSamples.proportion) / (1.f - nSamples.proportion);

            // Choose the BxDF to be used for the current interaction.
            /*
            BxDF *bxdf = ChooseBxDF(*isect.bsdf, &pdf, u[0], &flags);
             */
            BxDF *bxdf = isect.bsdf->ChooseBxDF(&pdf, u[0], &flags);

            if (bxdf == nullptr)
                // FIXME Handle that correctly (see path integrator).
                break;

            flags = bxdf->type;
            specularBounce = (flags & BSDF_SPECULAR) != 0;

            Vector3f localWo = isect.bsdf->WorldToLocal(wo);
            Vector3f localWi;

            if (localWo.z == 0) {
                break;
            }

            // If the interaction is specular, it is necessary to sample from
            // the BSDF.
            if (flags & BSDF_SPECULAR) {
                f = bxdf->Sample_f(localWo, &localWi, u, &pdfBsdf, &flags);
                wi = isect.bsdf->LocalToWorld(localWi);
                pdfL = samplingTree->Pdf(isect.p, wi);
                //misW = misPower(pdfBsdf, pdfL);
            } else {
                wi = samplingTree->Sample_L(isect.p, isect.n, u, &pdfL);
                localWi = isect.bsdf->WorldToLocal(wi);

                f = isect.bsdf->f(wo, wi, flags);
                pdfBsdf = isect.bsdf->Pdf(wo, wi);
                pdfL = samplingTree->Pdf(isect.p, wi);
                //misW = misPower(pdfL, pdfBsdf);
            }
        }

        // Sample the BSDF.
        else {
            u.x /= nSamples.proportion;
            f = isect.bsdf->Sample_f(wo, &wi, u, &pdfBsdf, BSDF_ALL, &flags);
            pdfL = recordingTree->Pdf(isect.p, wi);
        }

        pdf = nSamples.proportion*pdfBsdf + (1.f-nSamples.proportion)*pdfL;

        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }

        ray = isect.SpawnRay(wi);

        if (f.IsBlack() || pdf < MachineEpsilon) {
            break;
        }

        // Update the throughput.                               v Check this...
        Spectrum vertexBeta = f * AbsDot(wi, isect.shading.n) * misW / pdf;
        beta *= vertexBeta;

        if (! std::isfinite(beta.y())) {
            LOG(ERROR) << "OMG an infinite beta " << vertexBeta << " " << f;
        }

        // And add the new vertex to the path.
        path.emplace_back(PathVertex {isect.p, wi, vertexBeta});


#if 0 // SSS
        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                    scene, sampler->Get1D(), sampler->Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component
            L += beta * UniformSampleOneLight(pi, scene, arena, tileSampler, false,
                                              lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component
            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler->Get2D(), &pdf,
                                           BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0) break;
            beta *= f * AbsDot(wi, pi.shading.n) / pdf;
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }
#endif

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (tileSampler.Get1D() < q)
                break;
            beta /= 1 - q;
            DCHECK(!isinf(beta.y()));
        }
    }

    ReportValue(pathLength, bounces);
    return L;
}


void SDGuidedIntegrator::RecordPath(const vector<PathVertex> &path,
                                    Spectrum L) {
    // Loop over the path in reverse order, store the light estimation, and
    // apply the beta attenuation.
#if 1
    for (auto v = path.rbegin(); v < path.rend(); ++ v) {
        recordingTree->Record(v->p, v->wi, L.y());
        L *= v->beta;
    }
#else
    if (path.size() == 0)
        return;

    auto v = path.end() - 1;
    recordingTree->Record(v->p, v->wi, L.y());
#endif
}


SDGuidedIntegrator* CreatePPGIntegrator(const ParamSet& params,
                                             shared_ptr<Sampler> sampler,
                                             shared_ptr<const Camera> camera) {
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
    string lightStrategy =
            params.FindOneString("lightsamplestrategy", "spatial");
    return new SDGuidedIntegrator(maxDepth, camera, sampler,
                                  pixelBounds, rrThreshold, lightStrategy);
}


}
