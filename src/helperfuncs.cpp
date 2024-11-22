

#include "helperfuncs.h"

void printMatrix(const DoubleVector2D& mat) {
    for (const std::vector<double>& row : mat) {
        for (const double& elem : row) {
            // Adjust the width for better alignment
            std::cout << std::setw(8) << elem << " ";
        }
        std::cout << std::endl; // New line after each row
    }
}

void sigmoid(DoubleVector2D& mat) {
    for (std::vector<double>& row : mat) {
        for (double& elem : row) {
            // Sigmoid on each elemement
            elem = 1.0 / (1.0 + std::exp(-elem)); 
        }
    }
}
void tanh(DoubleVector2D& mat) {
    for (std::vector<double>& row : mat) {
        for (double& elem : row) {
            // Tanh on each elemement
            elem = 2 / (1 + std::exp(-2*elem)) - 1;
        }
    }
}

DoubleVector2D matrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2) {

    // Ensure that dimensions are correct (mat1 cols == mat2 rows)
    if (mat1[0].size() != mat2.size()){
        throw std::invalid_argument("Number of columns in mat1 must be eqaul to the number of rows in mat2");
    }

    // Now perform matrix mult
    const size_t numOutpRows = mat1.size();
    const size_t numOutpCols = mat2[0].size();
    const size_t numInner = mat2.size();

    DoubleVector2D outpMat(numOutpRows, std::vector<double>(numOutpCols, 0.0));

    
    // Populate each element in `outpMat`
    for (size_t row = 0; row < numOutpRows; row++) {
        for (size_t col = 0; col < numOutpCols; col++) {

            // Calc element
            double runningSum = 0;
            for (size_t inner = 0; inner < numInner; inner++) {
                runningSum += mat1[row][inner] * mat2[inner][col];
            }
            // Populate element
            outpMat[row][col] = runningSum;
        }
    }
    return outpMat;
}

void printMlpEnum(MLP::InitMethod inpEnum) {
    switch(inpEnum) {
        case MLP::InitMethod::UNIFORM:
            std::cout << "UNIFORM" << std::endl;
            break;
        case MLP::InitMethod::GAUSSIAN:
            std::cout << "GAUSSIAN" << std::endl;
            break;
        default:
            std::cout << "Printing for this not implemented just yet, so get here and do it" << std::endl;
    }
    return;
}
void printMlpEnum(MLP::ActFunc inpEnum);
void printMlpEnum(MLP::LossFunc inpEnum);






 

