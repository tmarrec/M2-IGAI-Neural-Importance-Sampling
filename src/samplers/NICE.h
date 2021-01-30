#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

class NICE
{
public:
	NICE()
	{
		pybind11::exec(R"(
			a = 0
		)");
	}

	void get()
	{
		pybind11::exec(R"(
			print(a)
		)");
	}

	void send()
	{
		pybind11::exec(R"(
			a = a + 1
		)");
	}

private:
};
