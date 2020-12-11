//
// Created by hrens on 5/4/18.
//

#ifndef PBRT_V3_STORM_SDTREE_H
#define PBRT_V3_STORM_SDTREE_H

#include <array>
#include <vector>
#include <atomic>
#include <core/parallel.h>
#include <core/geometry.h>
#include <fstream>
#include <random>

//#define TREE_DEBUG // If this is uncommented, then meh :(

namespace mapping {

    /// Map a 3D direction to a point in [0,1)².
    pbrt::Point2f DirToPlane(const pbrt::Vector3f& d);

    /// Map a (canonical) point from [0,1)² to a 3D direction.
    pbrt::Vector3f PlaneToDir(const pbrt::Point2f& p);

};

using pbrt::AtomicFloat;


///
/// The Spatial-Directional Tree as described by Müller and al. [2017].
/// The structure is basically a 3d-tree whose leafs hold references to
/// quadtrees, representing directional distributions.
///
class SDTree {
public:

    // Some cool types. The structure must be able to be accessed by many
    // concurrent threads.
    using AtomicBool = std::atomic_bool;
    using AtomicSize = std::atomic_size_t;
    using NodeRef = size_t;

#ifdef TREE_DEBUG
    struct ObjOutput {
        std::ofstream tree;
        std::ofstream samples;

        ObjOutput(const std::string &objname)
        : tree{objname+"_tree.obj", std::ios_base::out|std::ios_base::trunc}
        , samples{objname+"_samples.obj", std::ios_base::out|std::ios_base::trunc}
        {}
    };
#endif

    ///
    /// Quad tree representing a directional distribution. The range of this
    /// tree is implicitly [0, 1)² because of the cylindrical mapping.
    ///
    struct DirectionalTree {

        /// Quadtree node.
        struct Node {
        private:
            static constexpr int childCount = 4;

        public:
            Node () {
                for (uint i = 0; i < childCount; ++ i) {
                    children[i] = 0;
                    means[i] = 0.f;
                }
            }

            Node (const Node& n) {
                for (uint i = 0; i < childCount; ++ i) {
                    children[i] = n.children[i];
                    means[i] = static_cast<pbrt::Float>(n.means[i]);
                }
            }

            std::array<NodeRef, childCount> children;
            std::array<AtomicFloat, childCount> means;

            bool IsLeaf(uint i) const { return children[i] == 0; }
        };

        DirectionalTree();
        DirectionalTree(const DirectionalTree& t);

        /// Record a new radiance value in the radiance distribution.
        void InsertInPlane(pbrt::Point2f pos, pbrt::Float value);

        /// Insert a record coming from incident direction wi.
        void InsertRecord(pbrt::Vector3f wi, pbrt::Float value);

        /// Refine the distribution according to an energy-based heuristic.
        /// Specifically, when a node contains more than 1% of the total energy
        /// it is subdivided recursively until every node respect this property.
        void Refine(const DirectionalTree& base);

        /// Get the probability density of sampling a specific direction.
        pbrt::Float Pdf(pbrt::Vector3f wi) const;

        /// Sample a 2D point from the distribution, with coordinates in the
        /// unit plane [0, 1)².
#define REC
#ifndef REC
        pbrt::Point2f SampleInPlane(pbrt::Point2f u, pbrt::Float *pdf) const;
#else
        pbrt::Point2f SampleInPlane(pbrt::Point2f u, NodeRef ref=0,
                                    pbrt::Float *pdf=nullptr) const;
#endif

        /// Sample a direction from the probability tree.
        /// @warning The pdf isn't valid as of today, one must use Pdf(...) to
        /// get the correct value.
        pbrt::Vector3f SampleDirection(pbrt::Point2f u, pbrt::Float *pdf) const;

#ifndef TREE_DEBUG
    private:
#endif
        /// Returns the right child 'local' index (in [0..4)) according to
        /// the position of the node.
        uint ChildFromPos(NodeRef ref, pbrt::Point2f& pos) const;

#ifdef TREE_DEBUG
        /// Return the bounding box of a child of any node with the given box.
        pbrt::Bounds2f SubBbox(pbrt::Bounds2f box, uint child) const;

        /// Plot the centers of each node projected onto a sphere in the bbox,
        /// and draw 1000 samples from the distribution.
        uint ToObj(ObjOutput& objfiles, pbrt::Bounds3f bbox) const;

        /// Compute the depth of the quadtree.
        uint Depth(const NodeRef ref=0) const;
#endif

        /// Take a point in [0,1)² and convert it so that the interval
        /// correspond to the space covered by the given child.
        pbrt::Point2f MapToChildSubdomain(pbrt::Point2f p, uint child) const;

#ifndef TREE_DEBUG
    private:
#endif
        /// Nodes of the quadtree are stored linearly. There is no need to
        /// remove nodes, making the management simpler.
        std::vector<Node> nodes;

        /// Maximum depth.
        uint maxDepth;

        /// Total flux.
        AtomicFloat sum;

    };

    ///
    /// 3d-tree representing a spatial distribution.
    /// This tree is specific as its cut policy is to split the space in half
    /// every time (and not to keep the same number of points on both parts).
    /// Also, no node will ever be removed, thus a simpler memory management
    /// is permitted.
    ///
    struct SpatialTree {

        /// Kd-tree node.
        struct Node {
            Node() : lower{0}, upper{0}, isLeaf{0}
            {
                counter = 0;
            }

            Node(const Node& other)
                : lower{other.lower}
                , upper{other.upper}
                , counter{static_cast<size_t>(other.counter)}
                , isLeaf{other.isLeaf}
            {}

            Node(NodeRef l, NodeRef u, size_t count, uint isLeaf)
                : lower{l}, upper{u}, isLeaf{isLeaf}
            {
                counter = count;
            }

            NodeRef lower;         ///< Also used as a quadtree ref. if leaf.
            NodeRef upper;         ///< Valid only in internal nodes.
            AtomicSize counter;    ///< Only valid in leaf.
            uint isLeaf;           ///< 0 if internal, != 0 if leaf.
        };

        /// The ctor by default doesn't subdivide.
        SpatialTree(const pbrt::Bounds3f& boundaries, uint maxDepth = 20);

        /// Subdivide the space represented by a node.
        /// @warning The function make the assumption that any splitting
        /// condition was checked by the caller.
        void Split(NodeRef ref);

        /// Refine the included spatial distributions (the leafs) and then the
        /// directional distribution.
        void Refine(SpatialTree& other, size_t lastIteration);

        /// Find the node containing a point.
        const DirectionalTree& Find(pbrt::Point3f& p) const;

        /// Find the node containing a point and increment the counter in the
        /// spatial tree node.
        DirectionalTree& FindAndIncr(pbrt::Point3f& p);

        /// Define the criterion used to split a node.
        bool MustSubdivideSpace(NodeRef ref, size_t iteration) const;

#ifdef TREE_DEBUG
        /// Export a spatial tree as .obj.
        void ToObj(ObjOutput& objfiles);

        /// Export one node of the spatial tree (recursive function).
        void NodeToObj(NodeRef ref, ObjOutput& objfiles, pbrt::Bounds3f bbox,
                       std::vector<size_t>& boxVertices, uint axis=0);
#else
    private:
#endif

        /// The actual number of dimensions.
        static constexpr uint k = 3;

        /// The spatial range of the tree.
        pbrt::Bounds3f boundaries;

        /// Nodes are stored in a linear structure and referred to via indices.
        std::vector<Node> nodes;

        /// Leafs all keep a reference index to a quadtree, that are stored
        /// here.
        std::vector<DirectionalTree> quadtrees;

        /// Maximum depth of the trees.
        uint maxDepth;

#ifdef TREE_DEBUG
        size_t verticesEmitted = 0;
#endif
    };

    using DTree = DirectionalTree;
    using STree = SpatialTree;

public:
    SDTree(pbrt::Bounds3f boundaries =
           {pbrt::Point3f(-BIG_DISTANCE, -BIG_DISTANCE, -BIG_DISTANCE),
            pbrt::Point3f( BIG_DISTANCE,  BIG_DISTANCE,  BIG_DISTANCE)});

    /// Refine the incident radiance field estimation. In practice this means
    /// refining all the directional estimations, and then creating new nodes
    /// in the spatial tree where needed (each child inheriting of a copy of
    /// the directional component (the leaf)).
    void Refine(SDTree &other, size_t lastIteration);

    /// Sample the incident radiance distribution at point p. The normal is
    /// likely to be unused but it may be useful, thus it will be removed in
    /// the future.
    pbrt::Vector3f Sample_L(pbrt::Point3f p, pbrt::Normal3f n,
                            pbrt::Point2f u, pbrt::Float *pdf) const;

    /// Record an incident radiance value in the structure.
    void Record(pbrt::Point3f p, pbrt::Vector3f wi, pbrt::Float value);

    /// Get the probability density of sampling a given direction at a given
    /// point in space.
    pbrt::Float Pdf(pbrt::Point3f p, pbrt::Vector3f wi) const;

#ifdef TREE_DEBUG
    size_t NodesCount() const { return tree.nodes.size(); }

    /// Export a SDTree to a couple of .obj files representing the structure
    /// and samples drawn from it.
    void ToObj(const std::string& objname);

    /// Test mappings
    void TestMappings(uint resolution,
                      pbrt::Float epsilon=pbrt::MachineEpsilon);
#else
private:
#endif
    /// Spatial tree.
    SpatialTree tree;

    static constexpr float BIG_DISTANCE = 100.f;

};


#include "sdtree.h"

#endif //PBRT_V3_STORM_SDTREE_H
