#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef std::vector<std::vector<float>> vec2D;
typedef std::vector<float> vec1D;

class NICE {
public:
	NICE() {
		// Import du module et de la classe NICE du Python
		py::exec(R"(
			import sys
			sys.path.insert(0, "NICE")
		)");
		_module = py::module_::import("nice");
		_NICE_class = _module.attr("Nice")(2, 2);
	}

	void learn(vec2D paths, vec1D probas) {
		auto paths_np = py::array_t<float>(py::cast(paths));
		auto probas_np = py::array_t<float>(py::cast(probas));
		_NICE_class.attr("learn")(paths_np, probas_np);
	}

	std::tuple<vec2D, vec1D> get_paths(unsigned int num_path) {
		// Generation des chemins par l'implementation Python de NICE
		py::object result;
		result = _NICE_class.attr("generate_paths")(num_path);
		py::tuple tuple_result = result.cast<py::tuple>();

		vec2D paths;
		vec1D probas;

		fill_paths(tuple_result, paths, probas);
		return std::tuple<vec2D, vec1D>(paths, probas);
	}

	void fill_paths(py::tuple& tuple, vec2D& paths, vec1D& probas) {
		// Conversion des ndarray numpy en C++ vector
		for (auto t : tuple)
		{
			for (auto elem : t.cast<py::array_t<float>>())
			{
				if (elem.cast<py::array_t<float>>().size() > 1)
				{
					vec1D path;
					for (auto e : elem)
						path.emplace_back(e.cast<float>());
					paths.emplace_back(path);
				}
				else
				{
					probas.emplace_back(elem.cast<float>());
				}
			}
		}
	}

private:
	py::module_ _module;
	py::object _NICE_class;
};
