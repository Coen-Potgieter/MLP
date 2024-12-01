
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include "alias.h"
#include "mlp.h"

namespace fs = std::filesystem;

void printMatrix(const DoubleVector2D& mat);
void sigmoid(DoubleVector2D& mat);
void tanh(DoubleVector2D& mat);
DoubleVector2D matrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2);
void printMlpEnum(MLP::InitMethod inpEnum);
void printMlpEnum(MLP::ActFunc inpEnum);
void printMlpEnum(MLP::LossFunc inpEnum);
std::vector<std::string> separateRow(const std::string& content);
DoubleVector2D importCSV(std::string_view pathToCSV);
void normaliseData(DoubleVector2D& data);
DoubleVector2D separateTarget(DoubleVector2D& data);
DoubleVector2D transpose(const DoubleVector2D& matrix);
DoubleVector2D sliceRows(const DoubleVector2D& inpMatrix, const size_t startIdx, const size_t endIdx);
void printData(DoubleVector2D data, const size_t& numRows=0);




