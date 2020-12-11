
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

#ifndef PBRT_ACCELERATORS_HDKDTREE_H
#define PBRT_ACCELERATORS_HDKDTREE_H

#define ACC_DISTRIB

// accelerators/hdkdtree.h*
#include "pbrt.h"
#include "geometry.h"
#include <algorithm>
#include <iomanip>

namespace pbrt {

template<unsigned D>
struct HDKdQuery {
	std::vector<Point2f> ranges;
	HDKdQuery() {
		ranges = std::vector<Point2f>(D, Point2f(0.f, 1.f));
	}
	HDKdQuery<D>& operator=(const HDKdQuery<D> &kdquery) {
		ranges = kdquery.ranges;
		return *this;
	}
	inline void splitAxis(const unsigned axis, bool isLeft) {
		CHECK_LT(axis, D);
		const Float mid = 0.5f * (ranges[axis].x + ranges[axis].y);
		if (isLeft) ranges[axis].y = mid;
		else		ranges[axis].x = mid;
	}
	inline bool compAxis(const unsigned axis, const Float v) {
		return v < 0.5f * (ranges[axis].x + ranges[axis].y);
	}
	void warpSamples(std::vector<Point2f> &u, std::vector<Float> &pdf) {
		auto lerp = [](float l, float r, float v) { return (r - l) * v + l; };
		for (size_t i = 0; i < D / 2; ++i) {
			// Transform samples
			u[i].x = lerp(ranges[i].x,   ranges[i].y,   u[i].x);
			u[i].y = lerp(ranges[i+1].x, ranges[i+1].y, u[i].y);
		}
	}
	inline std::string ToString() {
		std::stringstream ss;
		for (auto i : ranges) ss << "{ " << i << " } "; ss << std::endl;
		return ss.str();
	}
};

struct HDKdNode {
	// HDKdNodeNew member variables
	// bits 0-30: count/offset, bit 31: leaf flag
	union {
		unsigned count;	// bits 0-30: count of path records in this branch
		unsigned offset;// bits 0-30: offset into node array for left child
	};					// 4 Bytes
	Float value;		// 4 Bytes

	// HDKdNodeNew member functions
	// Default constructor
	HDKdNode() : value(0.f), count(0) {}
	// Default constructor: isLeaf, count/offset, value
	HDKdNode(bool isLeaf, unsigned _count, Float _value)
		: value(_value) {
		count = (_count & 0x7fffffff) | (int(isLeaf) << 31);
	}
	inline bool isLeaf() { return (count & ~0x7ffffff) > 0; }
	inline unsigned getCount() { return count & 0x7ffffff; }
	inline unsigned getOffset() { return offset & 0x7ffffff; }
	inline void setCount(const unsigned _count) { // made leaf
		count = (_count & 0x7fffffff) | (0b1 << 31);
	}
	inline void setOffset(const unsigned _offset) { // made inner
		offset = (_offset & 0x7fffffff) | (0b0 << 31);
	}
	inline void addCount() { assert(isLeaf()); setCount(getCount() + 1); }
	inline void addValue(const Float _value) { value += _value; }
	inline std::string ToString() {
		std::stringstream ss;
		if (isLeaf())	ss << "[Leaf] - count: " << getCount();
		else			ss << "[Inner]-offset: " << getOffset();
		ss << " - value: " << value << std::endl;
		return ss.str();
	}
}; // 8 Bytes :)

template<unsigned D>
struct PathRecord {
	Float u[D];			// total D dimensions
	Float value;		// path luminance, L.y()
	unsigned dim;		// valid dim of current path record
	PathRecord(std::vector<Point2f> &u2, Spectrum &L) {
		// ideally D = 2 * u2.size()
		dim = std::min((int)(D * 0.5f), (int)u2.size());
		for (size_t i = 0; i < dim; ++i) {
			u[2 * i]	 = u2[i].x;
			u[2 * i + 1] = u2[i].y;
		}
		value = L.y();
	}
	inline std::string ToString() {
		std::stringstream ss;
		ss << "{ "; for (auto i : u) ss << i << " ";
		ss << "}*" << std::setprecision(16) << value;
		return ss.str();
	}
};

template<unsigned D>
class HDKdTree {
public:
	// HDKdTree public methods
	HDKdTree(unsigned mInLeaf = 4, Float initV = 10.f, unsigned mDepth = 20)
		: m_ndDefVal(initV), m_maxDepth(mDepth), m_maxPntInLeaf(mInLeaf) {
		if (D > m_maxDepth)
			throw std::runtime_error("D of HDKdTree<D> out of range!");
		m_maxDepth = (m_maxDepth / D) * D;
		initTree();
	}
	~HDKdTree() { m_root.clear(); m_collect.clear(); }

	inline void AddPathRecord(const PathRecord<D> &pr) {
		// Call the recursive traversal for collecting
		HDKdQuery<D> ranges;
		traverse(0, 0, pr, ranges);
	}
	inline void Iterate() {
		// Copy m_collect to m_root
		m_root = std::vector<HDKdNode>(m_collect);
#ifndef ACC_DISTRIB
		// Make a new _m_collect_ from scratch after each iteration
		m_collect.clear();
		initTree();
#endif	// ACC_DISTRIB
	}
	// Transform input samples by sampling the KdTree
	inline void SampleTree(	Sampler &s, std::vector<Point2f> &u,
							std::vector<Float> &pdf) {
		std::fill(pdf.begin(), pdf.end(), 1.f);
		if (!(m_root[0].value > 0.f))	return;
		HDKdQuery<D> ranges;
		// Recursively find a _ranges_ and update _pdf_ accordingly
		traverse(s, 0, 0, ranges, pdf);
		// Transform samples using _ranges_
		warpSamples(ranges, u, pdf);
	}
	// Get pdfs for MIS weighting by sampling the KdTree
	inline void SampleTree(	const std::vector<Point2f> &u,
							std::vector<Float> &pdf) {
		std::fill(pdf.begin(), pdf.end(), 1.f);
		if (!(m_root[0].value > 0.f))	return;
		HDKdQuery<D> ranges;
		// Recursively locate _u_ in the tree and update pdf
		traverse(0, 0, u, ranges, pdf);
	}
	inline std::string ToString() {
		std::stringstream ss;
		for (size_t i = 0; i < m_root.size(); ++i)
			ss << "[" << i << "]" << m_root[i].ToString();
		return ss.str();
	}

private:
	// HDKdTree private methods
	// Build initial tree
	void initTree() {
		// Make a root node and start recursive build
		m_collect.reserve(int(std::pow(2.f, D + 1)));
		Float rootValue = m_ndDefVal * std::pow(2.f, D);
		HDKdNode root = HDKdNode();
		m_collect.push_back(root);
		buildTree0(0, 0);
		Iterate();
	}
	void buildTree0(const unsigned index, const unsigned depth) {
		CHECK_LT(depth, m_maxDepth);
		
		// Update content of current node
		const bool isLeaf = depth >= D;
		const Float value = m_ndDefVal * std::pow(2.f, D - depth);
		const unsigned leftIdxOrCnt = (unsigned)m_collect.size();
		m_collect[index].value = value;
		// Possibly terminate or make two children
		if (isLeaf) {
			m_collect[index].setCount(0);
			return;
		} else {
			m_collect[index].setOffset(leftIdxOrCnt);
			HDKdNode firstChild, secondChild;
			m_collect.push_back(firstChild );
			m_collect.push_back(secondChild);
			buildTree0(leftIdxOrCnt,	 depth + 1);
			buildTree0(leftIdxOrCnt + 1, depth + 1);
		}
	}
	// the recursive traversal method with PathRecord, for collecting
	void traverse(const unsigned index, const unsigned depth,
				  const PathRecord<D> &pr, HDKdQuery<D> ranges) {
		CHECK_LT((size_t)index, m_collect.size());
		CHECK_LT(depth, m_maxDepth);

		const unsigned axis = depth % D;
		const Float sample = pr.u[axis];
		const Float value = pr.value > 0.f ? pr.value : 0.f;
		const bool goLeft = ranges.compAxis(axis, sample);

		if (!m_collect[index].isLeaf()) {	
			// Add value to all traversed inner nodes
			m_collect[index].addValue(value);
			// Compare to check which direction to go
			HDKdQuery<D> childRanges = ranges;
			childRanges.splitAxis(axis, goLeft);
			const unsigned childIndex = m_collect[index].getOffset();
			const unsigned nextIndex = childIndex + (goLeft ? 0 : 1);
			traverse(nextIndex, depth + 1, pr, childRanges);
		} else {
			const Float oldValue = 0.5f * m_collect[index].value;
			// Leaf nodes: add count and value
			m_collect[index].addCount();
			m_collect[index].addValue(value);
			// possibly make current node inner and make new leaf nodes
			if (m_collect[index].getCount() > m_maxPntInLeaf &&
				depth < m_maxDepth - 1) {
				const unsigned halfPnt = m_maxPntInLeaf / 2;
				const unsigned lCount  = goLeft ? halfPnt + 1 : halfPnt;
				const unsigned rCount  = goLeft ? halfPnt : halfPnt + 1;
				const unsigned leftInx = (unsigned)m_collect.size();
				const Float lValue = goLeft ? oldValue + value : oldValue;
				const Float rValue = goLeft ? oldValue : oldValue + value;
				// Make two slightly different nodes with current input
				HDKdNode firstChild  = HDKdNode(true, lCount, lValue);
				HDKdNode secondChild = HDKdNode(true, rCount, rValue);
				m_collect.push_back(firstChild );
				m_collect.push_back(secondChild);
				// Make current node inner
				m_collect[index].setOffset(leftInx);
			}
		}
	}
	// the recursive traversal method with samples, for sampling
	void traverse(Sampler &s, const unsigned index, const unsigned depth,
				  HDKdQuery<D> &ranges, std::vector<Float>& pdf ) {
		CHECK_LT(depth, m_maxDepth);
		CHECK_LT((size_t)index, m_root.size());
		// Return if leaf node
		if (m_root[index].isLeaf())	return;
		// Sampling a child node according to _value_
		const Float currentValue = m_root[index].value;
		CHECK_GT(currentValue, 0.f);
		const unsigned leftIndex = m_root[index].getOffset();
		CHECK_LT(leftIndex, m_root.size());
		const Float leftValue = m_root[leftIndex].value;
		const Float leftNormProb = leftValue / currentValue;
		const bool goLeft = s.Get1D() < leftNormProb;
		const unsigned nextIndex = goLeft ? leftIndex : leftIndex + 1;
		const unsigned axis = depth % D;
		ranges.splitAxis(axis, goLeft);
		pdf[axis] *= (goLeft ? leftNormProb : (1.f - leftNormProb)) * 2.f;
		traverse(s, nextIndex, depth + 1, ranges, pdf);
	}
	// the recursive traversal method with samples, for MIS sampling
	void traverse(const unsigned idx, const unsigned depth, 
				  const std::vector<Point2f> &u, 
				  HDKdQuery<D> &ranges, std::vector<Float> &pdf) {
		CHECK_LT(depth, m_maxDepth);
		CHECK_LT((size_t)idx, m_root.size());
		// Return if leaf node
		if (m_root[idx].isLeaf())	return;
		const unsigned axis = depth % D;
		const Float currentValue = m_root[idx].value;
		CHECK_GT(currentValue, 0.f);
		// Go left or right by comparing entry value in _u_
		const bool goLeft = ranges.compAxis(axis, u[axis/2][axis%2]);
		const unsigned leftIndex = m_root[idx].getOffset();
		const Float leftValue = m_root[leftIndex].value;
		const Float leftNormProb = leftValue / currentValue;
		// Update _ranges_ and _pdf_
		ranges.splitAxis(axis, goLeft);
		pdf[axis] *= (goLeft ? leftNormProb : (1.f - leftNormProb)) * 2.f;
		const unsigned nextIndex = goLeft ? leftIndex : leftIndex + 1;
		traverse(nextIndex, depth + 1, u, ranges, pdf);
	}
	// Warp samples in _u_ using a queried range _r_
	inline void warpSamples(HDKdQuery<D> &r, std::vector<Point2f> &u,
							std::vector<Float> &pdf) {
		if (2 * u.size() != D) return;
		r.warpSamples(u, pdf);
	}

private:
	// HDKdTree private data
	const Float m_ndDefVal;
	std::vector<HDKdNode> m_root, m_collect;
	unsigned m_maxDepth, m_maxPntInLeaf;
};

}	// namespace pbrt
#endif	// PBRT_ACCELERATORS_HDKDTREE_H
