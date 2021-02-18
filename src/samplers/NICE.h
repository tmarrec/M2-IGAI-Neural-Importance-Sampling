#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef std::vector<std::vector<double>> d2D;
typedef std::vector<double> d1D;

class NICE
{
public:
	NICE()
	{
		// Import du module et de la classe NICE du Python
		_module = py::module_::import("dummy");
		_NICE_class = _module.attr("NiceDummy").call(2, 2);
	}

	void learn(d2D paths, d1D probas) {
		auto paths_np = py::array_t<double>(py::cast(paths));
		auto probas_np = py::array_t<double>(py::cast(probas));
	}

	std::tuple<d2D, d1D> get_paths(unsigned int num_path)
	{
		// Generation des chemins par l'implementation Python de NICE
		py::object result;
		py::tuple tuple_result = result.cast<py::tuple>();

		d2D paths;
		d1D probas;
		
		fill_paths(tuple_result, paths, probas);

		return std::tuple<d2D, d1D>(paths, probas);
	}

	void fill_paths(py::tuple& tuple, d2D& paths, d1D& probas)
	{
		// Conversion des ndarray numpy en C++ vector
		for (auto t : tuple)
		{
			for (auto elem : t.cast<py::array_t<double>>())
			{
				if (elem.cast<py::array_t<double>>().size() > 1)
				{
					d1D path;
					for (auto e : elem)
						path.emplace_back(e.cast<double>());
					paths.emplace_back(path);
				}
				else
				{
					probas.emplace_back(elem.cast<double>());
				}
			}
		}
	}

private:
	py::module_ _module;
	py::object _NICE_class;
};
