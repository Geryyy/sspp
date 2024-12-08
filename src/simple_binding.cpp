//
// Created by geraldebmer on 15.11.24.
//

#include <pybind11/pybind11.h>

// Define the function that adds two numbers
int add(int a, int b) {
    return a + b;
}

// Pybind11 module definition
PYBIND11_MODULE(sspp_bindings, m) {
    m.doc() = "A simple module that adds two numbers"; // Optional module docstring
    m.def("add", &add, "A function that adds two numbers",
          pybind11::arg("a"), pybind11::arg("b")); // Function name and signature
}
