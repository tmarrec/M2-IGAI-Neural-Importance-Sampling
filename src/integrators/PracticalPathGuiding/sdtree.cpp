//
// Created by hrens on 5/4/18.
//

#include "stats.h"
#include "sdtree.h"
#include "samplers/random.h"

#include <stack>
#include <random>
#include <iostream>
#include <cmath>

#include <ctime>

using namespace pbrt;
using namespace std;


Point2f mapping::DirToPlane(const pbrt::Vector3f &d) {
    if (!isfinite(d.x) || !isfinite(d.y) || !isfinite(d.z))
        return {0, 0};

    const Float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
    Float phi = std::atan2(d.y, d.x);

    while (phi < 0)
        phi += 2.0 * M_PI;

    return {(cosTheta + 1) / 2, phi / (2 * (Float)M_PI)};
}


Vector3f mapping::PlaneToDir(const pbrt::Point2f &p) {
    const Float cosTheta = 2*p.x - 1;
    const Float phi = 2 * (Float)M_PI * p.y;

    const Float sinTheta = sqrt(1 - (cosTheta*cosTheta));
    const Float sinPhi = sin(phi);
    const Float cosPhi = cos(phi);

    return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}


//
// Directional tree
//

SDTree::DirectionalTree::DirectionalTree()
    : maxDepth{20}
    , sum{0} {
    nodes.emplace_back();
}


SDTree::DirectionalTree::DirectionalTree(const DirectionalTree& t)
    : nodes{t.nodes}
    , maxDepth{t.maxDepth}
    , sum(static_cast<Float>(t.sum)) {
    // Nothing.
}


void SDTree::DirectionalTree::InsertInPlane(Point2f pos, Float value) {
    NodeRef current = 0;
    uint child = ChildFromPos(current, pos);

    while (! nodes[current].IsLeaf(child)) {
        pos = MapToChildSubdomain(pos, child);
        nodes[current].means[child].Add(value);
        current = nodes[current].children[child];
        child = ChildFromPos(current, pos);
    }

    nodes[current].means[child].Add(value);
    sum.Add(value);
}


void SDTree::DirectionalTree::InsertRecord(Vector3f wi, Float value) {
    InsertInPlane(mapping::DirToPlane(wi), value);
}


void SDTree::DirectionalTree::Refine(const DirectionalTree& baseTree) {
    sum = 0;
    nodes.clear();
    nodes.emplace_back();

    // The underlying algorithm is taken from the authors' code.
    // It uses a stack containing the various nodes to be inserted associated
    // with the node they must use as reference.

    const Float total = baseTree.sum;
    const uint newMaxDepth = maxDepth;

    maxDepth = 0;

    struct StackNode {
        NodeRef nodeIndex;
        NodeRef otherNodeIndex;
        const DirectionalTree* otherTree;
        uint depth;
    };

    stack<StackNode> nodeIndices;
    nodeIndices.push({0, 0, &baseTree, 1});

    while (! nodeIndices.empty()) {
        StackNode current = nodeIndices.top();
        nodeIndices.pop();

        const DirectionalTree::Node& otherNode =
                current.otherTree->nodes[current.otherNodeIndex];

        maxDepth = max(maxDepth, current.depth);

        // Explore each direct child.
        for (uint child = 0; child < 4; ++ child) {
            const Float nodeFraction = (total > 0) ?
                                       otherNode.means[child] / total : 0;

            if (current.depth < newMaxDepth && nodeFraction > 0.01) {
                if (! otherNode.IsLeaf(child)) {
                    nodeIndices.push({nodes.size(), otherNode.children[child],
                                      &baseTree, current.depth + 1});
                } else {
                    nodeIndices.push({nodes.size(), nodes.size(),
                                      this, current.depth + 1});
                }

                nodes[current.nodeIndex].children[child] =
                        static_cast<NodeRef>(nodes.size());

                nodes.emplace_back();

                const float distributedMean = (Float) otherNode.means[child]/4;
                for (uint i = 0; i < 4; ++ i)
                    nodes.back().means[i] = distributedMean;

                if (nodes.size() > numeric_limits<NodeRef>::max()) {
                    nodeIndices = stack<StackNode>();
                    cerr << "Maximum children count reached" << endl;
                    break;
                }
            }
        }
    }

    // Remove all energy from the new tree.
    for (auto& node: nodes) {
        for (uint child = 0; child < 4; ++ child) {
            node.means[child] = 0;
        }
    }
}


Float SDTree::DirectionalTree::Pdf(Vector3f wi) const {
    Point2f p = mapping::DirToPlane(wi);
    Float pdf = 1.f;

    // Descend the tree and update the pdf.
    NodeRef currentRef = 0;
    const Node* current = &nodes[currentRef];
    uint child;

    bool atLeaf;

    do {
        child = ChildFromPos(currentRef, p);

        Float total = current->means[0]+current->means[1]+
                      current->means[2]+current->means[3];

        if (total <= 0) {
            return 1 / (4.f*(Float)M_PI);
        }

        Float partial = current->means[child];

        if (partial <= 0) {
            return 0;
        }

        pdf *= 4*partial/total;
        atLeaf = current->IsLeaf(child);

        if (! atLeaf) {
            p = MapToChildSubdomain(p, child);
            currentRef = current->children[child];
            current = &nodes[currentRef];
        }

    } while (! atLeaf);

    return pdf / (4.f*(Float)M_PI);
}


#ifndef REC
Point2f SDTree::DirectionalTree::SampleInPlane(Point2f u, Float *pdf) const {
    const Node* current = &nodes[0];
    uint child;
    *pdf = 1.f;

    // This code is also mostly taken from the authors' code, however in a non
    // recursive form. It also compute the pdf online (it tries to, at least).

    float scale = 1.f;
    Point2f origin {0, 0};
    Vector2f boundary {0.5, 0.5};

    bool atLeaf;

    // At each step we descend a level in the tree, and maintain the origin of
    // the node, its scale (to scale the u accordingly) and the boundary (which
    // describes on x and y the separator between the two regions).
    //
    // The dimension x is always checked before the y one. The u sample is
    // scaled at each iteration to span on [0, 1).
    do {
        child = 0;

        float partial = current->means[0] + current->means[2];
        float total   = current->means[1] + current->means[3] + partial;

        if (total == 0)
            return u;

        boundary.x = partial / total;

        if (u.x < boundary.x) {
            u.x /= boundary.x;
            boundary.y = current->means[2] / partial;
            *pdf *= boundary.x;
        } else {
            partial = total - partial;
            origin.x += 0.5f * scale;
            u.x = (u.x - boundary.x) / (1.0f - boundary.x);
            boundary.y = current->means[3] / partial;
            child += 1;
            *pdf *= 1 - boundary.x;
        }

        if (u.y < boundary.y) {
            u.y /= boundary.y;
            child += 2;
            *pdf *= boundary.y;
        } else {
            u.y = (u.y - boundary.y) / (1.0f - boundary.y);
            origin.y += 0.5 * scale;
            *pdf *= 1 - boundary.y;
        }

        *pdf *= 4;

        atLeaf = current->IsLeaf(child);

        scale *= 0.5f;
        current = &nodes[current->children[child]];
    }
    while (! atLeaf);

    // FIXME WARNING The pdf isn't valid yet. Use the dedicated function Pdf()
    // for now.
    *pdf /= (4*M_PI);

    // In the end the point lie in the plane described by (origin, scale).
    return origin + u * scale;
}
#else
Point2f SDTree::DirectionalTree::SampleInPlane(Point2f u, NodeRef ref,
                                               Float *pdf) const {
    // Select a child according to its pdf and recursively sample the child.

    const Node& current = nodes[ref];
    uint child = 0;

    *pdf = 1/(4*(Float)M_PI);

    Point2f offset(0, 0);

    float partial = current.means[0] + current.means[2];
    float total   = current.means[1] + current.means[3] + partial;

    if (total <= 0) {
        bool deadend = true;
        for (uint i = 0; i < 4; ++ i)
            deadend = deadend && current.IsLeaf(i);
        if (deadend)
            return u;
    }

    float boundary = partial / total;

    // Chose child on x axis.
    if (u.x < boundary) {
        u.x /= boundary;
        boundary = (partial > 0) ? current.means[2] / partial : 0;
    } else {
        u.x = (u.x - boundary) / (1 - boundary);
        boundary = (partial < total) ? current.means[3] / (total-partial) : 0;
        offset.x = 0.5;
        child += 1;
    }

    // Chose child on y axis.
    if (u.y < boundary) {
        u.y /= boundary;
        child += 2;
    } else {
        u.y = (u.y - boundary) / (1 - boundary);
        offset.y = 0.5;
    }

    // Return either the position in the leaf or the recursion result.
    if (current.IsLeaf(child)) {
        return offset + 0.5*u;
    } else {
        return offset + 0.5*SampleInPlane(u, current.children[child], pdf);
    }
}
#endif


Vector3f SDTree::DirectionalTree::SampleDirection(Point2f u, Float *pdf) const {
#ifndef REC
    return mapping::PlaneToDir(SampleInPlane(u, pdf));
#else
    return mapping::PlaneToDir(SampleInPlane(u, 0, pdf));
#endif
}


uint SDTree::DirectionalTree::ChildFromPos(NodeRef ref, Point2f &pos) const {
    if (pos.y > 0.5) {
        if (pos.x <= 0.5)
            return 0;
        else
            return 1;
    } else {
        if (pos.x <= 0.5)
            return 2;
        else
            return 3;
    }
}


#ifdef TREE_DEBUG
Bounds2f SDTree::DirectionalTree::SubBbox(Bounds2f box, uint child) const {
    Point2f c = (box.pMin + box.pMax) * 0.5;

    switch (child) {
        case 0:
            return {Point2f(box.pMin.x, c.y), Point2f(c.x, box.pMax.y)};
        case 1:
            return {c, box.pMax};
        case 2:
            return {box.pMin, c};
        case 3:
            return {Point2f(c.x, box.pMin.y), Point2f(box.pMax.x, c.y)};
        default:
            cerr << "Invalid child " << child << endl;
            exit(EINVAL);
    }
}


uint SDTree::DirectionalTree::ToObj(ObjOutput& objfiles, pbrt::Bounds3f bbox)
    const {
    // Find span and center of the bounding box.
    Point3f c;
    c.x = (bbox[0].x + bbox[1].x) * 0.5f;
    c.y = (bbox[0].y + bbox[1].y) * 0.5f;
    c.z = (bbox[0].z + bbox[1].z) * 0.5f;

    Float span = std::min(std::min(bbox[1].x-bbox[0].x, bbox[1].y-bbox[0].y),
                           bbox[1].z-bbox[0].z) * (Float)0.9;

    auto plotPoint = [span](Vector3f dir, Point3f origin, ofstream& objfile) {
        auto p = origin + (.5f*span) * .8f * dir;
        objfile << "v " << p.x << " " << p.y << " " << p.z << endl;
    };

    // Scatter plot nodes centers.
    struct Candidate {
        Bounds2f box;
        NodeRef  ref;
    };

    stack<Candidate> candidates;
    auto domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
    candidates.push({domain, 0});

    uint count = 0;
    while (! candidates.empty()) {
        Candidate current = candidates.top();
        candidates.pop();

        for (uint child = 0; child < 4; ++ child) {
            auto childBox = SubBbox(current.box, child);
            if (nodes[current.ref].IsLeaf(child)) {
                auto center = childBox[0] + 0.5 * (childBox[1] - childBox[0]);
                auto dir = mapping::PlaneToDir(center);
                plotPoint(dir, c, objfiles.tree);
                ++ count;
            } else {
                candidates.push({childBox, nodes[current.ref].children[child]});
            }
        }
    }

    // Sample the sphere.
    count = 1000;
    RandomSampler sampler(2*count, static_cast<int>(time(nullptr)));
    float dull;
    // Zone 0.
    for (uint i = 0; i < count/4; ++ i) {
        auto uv  = sampler.Get2D() * 0.5 + Point2f(0, 0.5);
        auto dir = SampleDirection(uv, &dull);
        plotPoint(dir, c, objfiles.samples);
    }
    // Zone 1.
    for (uint i = 0; i < count/4; ++ i) {
        auto uv  = sampler.Get2D() * 0.5 + Point2f(0.5, 0.5);
        auto dir = SampleDirection(uv, &dull);
        plotPoint(dir, c, objfiles.samples);
    }
    // Zone 2.
    for (uint i = 0; i < count/4; ++ i) {
        auto uv  = sampler.Get2D() * 0.5;
        auto dir = SampleDirection(uv, &dull);
        plotPoint(dir, c, objfiles.samples);
    }
    // Zone 3.
    for (uint i = 0; i < count/4; ++ i) {
        auto uv  = sampler.Get2D() * 0.5 + Point2f(0.5, 0);
        auto dir = SampleDirection(uv, &dull);
        plotPoint(dir, c, objfiles.samples);
    }
}


uint SDTree::DirectionalTree::Depth(const NodeRef ref) const {
    const Node& node = nodes[ref];

    uint depths[4];
    depths[0] = (node.IsLeaf(0)) ? 0 : Depth(node.children[0]);
    depths[1] = (node.IsLeaf(1)) ? 0 : Depth(node.children[1]);
    depths[2] = (node.IsLeaf(2)) ? 0 : Depth(node.children[2]);
    depths[3] = (node.IsLeaf(3)) ? 0 : Depth(node.children[3]);

    return 1 + max(max(depths[0], depths[1]), max(depths[2], depths[3]));
}
#endif


Point2f SDTree::DirectionalTree::MapToChildSubdomain(Point2f pos, uint child)
const {
    if (child == 1) {
        pos.x -= 0.5;
        pos.y -= 0.5;
    } else if (child == 0) {
        pos.y -= 0.5;
    } else if (child == 3) {
        pos.x -= 0.5;
    }
    return pos * 2;
}


//
// Spatial tree
//

SDTree::SpatialTree::SpatialTree(const pbrt::Bounds3f& boundaries, uint maxDepth)
    : boundaries{boundaries}
    , maxDepth{maxDepth} {
    // Create an original node.
    quadtrees.emplace_back();
    nodes.emplace_back();
    nodes.back().upper = 0;
    nodes.back().lower = 0;
    nodes.back().isLeaf = 1;
    nodes.back().counter = 0;
}


void SDTree::SpatialTree::Split(SDTree::NodeRef ref) {
    if (! nodes[ref].isLeaf) {
        return;
    }

    // Create two new nodes and distribute counters values.
    NodeRef children = nodes.size();
    nodes.emplace_back(0, 0, (nodes[ref].counter/2) + nodes[ref].counter%2, 1);
    nodes.emplace_back(0, 0, (nodes[ref].counter/2), 1);

    // Duplicate the directional distributions over the new leafs.
    nodes[children  ].lower = nodes[ref].lower;
    nodes[children+1].lower = quadtrees.size();
    quadtrees.emplace_back(DirectionalTree(quadtrees[nodes[ref].lower]));

    // Update the current so that it is no longer a leaf.
    nodes[ref].isLeaf = 0;
    nodes[ref].lower = children;
    nodes[ref].upper = children+1;
}


void SDTree::SpatialTree::Refine(SDTree::SpatialTree &other,
                                 size_t lastIteration) {
    nodes.clear();
    for (auto const& onode: other.nodes) {
        nodes.emplace_back(onode);
    }

    // Refine directional distributions in parallel.
    quadtrees.resize(other.quadtrees.size());
    pbrt::ParallelFor([this, &other](int64_t leaf) {
        this->quadtrees[leaf].Refine(other.quadtrees[leaf]);
    }, other.quadtrees.size());

    // And then refine the spatial component.
    struct UnexploredNode {
        NodeRef index;
        uint depth;
    };

    stack<UnexploredNode> unexplored;
    unexplored.push({0, 1});

    // At each unexplored node, if it's a leaf then it is potentially to be
    // subdivided. Otherwise we just add its two for children to exploration.
    while (! unexplored.empty()) {
        UnexploredNode current = unexplored.top();
        unexplored.pop();

        if (nodes[current.index].isLeaf) {
            if (other.MustSubdivideSpace(current.index, lastIteration))
                Split(current.index);
        } else {
            unexplored.push({nodes[current.index].lower, current.depth+1});
            unexplored.push({nodes[current.index].upper, current.depth+1});
        }
    }
}


SDTree::DirectionalTree& SDTree::SpatialTree::FindAndIncr(pbrt::Point3f& p) {
    NodeRef currentRef = 0;
    uint d = 0;

    pbrt::Bounds3f box = boundaries;

    while (! nodes[currentRef].isLeaf) {
        const float middle = (box.pMax[d] + box.pMin[d])*0.5f;

        // Descend the tree and update the represented range.
        if (p[d] > middle) {
            currentRef = nodes[currentRef].upper;
            box.pMin[d] = middle;
        } else {
            currentRef = nodes[currentRef].lower;
            box.pMax[d] = middle;
        }

        d = (d+1) % k;
    }

    nodes[currentRef].counter += 1;

    return quadtrees[nodes[currentRef].lower];
}


const SDTree::DirectionalTree& SDTree::SpatialTree::Find(pbrt::Point3f& p)
        const {
    NodeRef currentRef = 0;
    uint d = 0;

    pbrt::Bounds3f box = boundaries;

    while (! nodes[currentRef].isLeaf) {
        const float middle = (box.pMax[d] + box.pMin[d])*0.5f;

        // Descend the tree and update the represented range.
        if (p[d] > middle) {
            currentRef = nodes[currentRef].upper;
            box.pMin[d] = middle;
        } else {
            currentRef = nodes[currentRef].lower;
            box.pMax[d] = middle;
        }

        d = (d+1) % k;
    }

    return quadtrees[nodes[currentRef].lower];
}


bool SDTree::SpatialTree::MustSubdivideSpace(SDTree::NodeRef ref,
                                             size_t lastIteration) const {
    if (ref >= nodes.size())
        return false;

    constexpr size_t resolution = 12000; // Value from the paper.
    return (static_cast<size_t>(nodes[ref].counter) >
            resolution*sqrt(pow(2, lastIteration)));
}


#ifdef TREE_DEBUG
void SDTree::SpatialTree::ToObj(ObjOutput& objfiles) {
    objfiles.tree    << "# .obj generated by a PBRT tool." << endl << endl;
    objfiles.samples << "# .obj generated by a PBRT tool." << endl << endl;

    // Big and ugly list of vertices that are part of the boxes.
    vector<size_t> bv;

    LOG(ERROR) << "Going to dump a tree of " << quadtrees.size() << " cells";

    // Descend in the nodes, and create vertices (while memorizing those who
    // are part of the boxes skeleton).
    NodeToObj(0, objfiles, boundaries, bv);
}


void SDTree::SpatialTree::NodeToObj(NodeRef ref, ObjOutput& objfiles,
                      pbrt::Bounds3f bbox, vector<size_t>& boxVertices,
                      uint axis) {
    if (nodes[ref].isLeaf) {
        // 1. Add the bounding box.
        for (uint mask = 0; mask < 8; ++ mask) {
            Point3f corner;
            corner.x = (mask & (1u << 0u)) ? bbox[1].x : bbox[0].x;
            corner.y = (mask & (1u << 1u)) ? bbox[1].y : bbox[0].y;
            corner.z = (mask & (1u << 2u)) ? bbox[1].z : bbox[0].z;
            boxVertices.push_back(verticesEmitted+1);
            objfiles.tree << "v " << corner.x <<
                              " " << corner.y << " " << corner.z << endl;
            ++ verticesEmitted;
        }
        // 2. And dump the quadtree.
        verticesEmitted += quadtrees[nodes[ref].lower].ToObj(objfiles, bbox);
    } else {
        // Find the sub-bounding boxes.
        Bounds3f lowerBbox = bbox;
        Bounds3f upperBbox = bbox;

        float span = bbox[1][axis] - bbox[0][axis];
        lowerBbox[1][axis] = lowerBbox[0][axis] + span/2;
        upperBbox[0][axis] = upperBbox[1][axis] - span/2;

        // And recursively descend.
        uint nextAxis = (axis+1)%3;
        NodeToObj(nodes[ref].lower, objfiles, lowerBbox, boxVertices, nextAxis);
        NodeToObj(nodes[ref].upper, objfiles, upperBbox, boxVertices, nextAxis);
    }
}
#endif


//
// Hybrid tree
//

SDTree::SDTree(Bounds3f boundaries)
    : tree{boundaries} {
    // Nothing.
}


void SDTree::Refine(SDTree &other, size_t lastIteration) {
    tree.Refine(other.tree, lastIteration);
}


Vector3f SDTree::Sample_L(Point3f p, Normal3f n, Point2f u, Float *pdf) const {
    // Find and sample the correct directional distribution.
    const DTree& localDistribution = tree.Find(p);
    Vector3f direction = localDistribution.SampleDirection(u, pdf);

    return direction;
}


void SDTree::Record(Point3f p, Vector3f wi, Float value) {
    DTree& localDistribution = tree.FindAndIncr(p);
    localDistribution.InsertRecord(wi, value);
}


Float SDTree::Pdf(Point3f p, Vector3f wi) const {
    const DTree& localDistribution = tree.Find(p);
    return localDistribution.Pdf(wi);
}


#ifdef TREE_DEBUG
void SDTree::ToObj(const std::string &objname) {
    ObjOutput objs {objname};
    tree.ToObj(objs);
}


void SDTree::TestMappings(uint resolution, Float epsilon) {
    uint total = resolution*resolution;
    uint failures = 0;

    // UV -> spherical -> UV
    for (uint u = 0; u < resolution; ++ u) {
        for (uint v = 0; v < resolution; ++ v) {
            auto fu = u / (Float) resolution;
            auto fv = v / (Float) resolution;
            auto fp = Point2f(fu, fv);

            auto p = mapping::DirToPlane(mapping::PlaneToDir(fp));

            if ((abs(p.x - fu) > epsilon) || (abs(p.y - fv) > epsilon)) {
                ++ failures;
                LOG(ERROR) << fp << " vs " << p;
            }
        }
    }

    LOG(ERROR) << "Mappings failures ratio: " << (100.f*failures)/(Float)total
               << "% (" << failures << "/" << total << ")";

    // spherical -> UV -> spherical
}
#endif
