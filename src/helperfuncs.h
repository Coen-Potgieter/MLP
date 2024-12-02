
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include "alias.h"
#include "mlp.h"

namespace fs = std::filesystem;

void printMatrix(const DoubleVector2D& mat);
void printDims(const DoubleVector2D& mat);

// Activations
double relu(const double& z);
double tanh(const double& z);
double derivativeRelu(const double& z);
double derivativeTanh(const double& z);
DoubleVector2D matrixActivation(const DoubleVector2D& Z);
DoubleVector2D matrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2);
DoubleVector2D elementWiseMatrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2);
void printMlpEnum(MLP::InitMethod inpEnum);
void printMlpEnum(MLP::ActFunc inpEnum);
void printMlpEnum(MLP::LossFunc inpEnum);
std::vector<std::string> separateRow(const std::string& content);
DoubleVector2D importCSV(std::string_view pathToCSV);
void normaliseData(DoubleVector2D& data);
DoubleVector2D separateTarget(DoubleVector2D& data, std::vector<int>& targetCols);
DoubleVector2D transpose(const DoubleVector2D& matrix);
DoubleVector2D sliceCols(const DoubleVector2D& inpMatrix, const size_t startIdx, const size_t endIdx);
void printData(DoubleVector2D data, const size_t& numRows=0);




