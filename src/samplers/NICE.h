#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

class NICE
{
public:
	NICE()
	{
	}

	void get()
	{
		py::scoped_interpreter interp;
		std::cout << "Je demande a NICE un truc en python" << std::endl;
		py::print("Bonjour je suis le python");
	}

	void send()
	{

	}

private:
};
