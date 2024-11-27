
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
std::vector<std::string> splitString(const std::string& content, char delimiter);
StringVector2D importCSV(std::string_view pathToCSV);
void printData(StringVector2D data, const size_t& numRows);



