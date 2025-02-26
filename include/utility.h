//
// Created by geraldebmer on 26.02.25.
//

#ifndef SSPP_UTILITY_H
#define SSPP_UTILITY_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

void exportToCSV(const std::string& filename, const Eigen::VectorXd& x, const Eigen::MatrixXd& y) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write header
    file << "x";
    for (int j = 0; j < y.rows(); ++j) {
        file << ", y" << j;  // Column names: y0, y1, ...
    }
    file << "\n";

    // Write data
    for (int i = 0; i < x.size(); ++i) {
        file << x(i); // X value
        for (int j = 0; j < y.rows(); ++j) {
            file << ", " << y(j, i); // Y values
        }
        file << "\n";
    }

    file.close();
    std::cout << "Data exported to " << filename << std::endl;
}


#endif //SSPP_UTILITY_H
