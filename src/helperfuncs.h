
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include "alias.h"
#include "mlp.h"

void printMatrix(const DoubleVector2D& mat);
void sigmoid(DoubleVector2D& mat);
void tanh(DoubleVector2D& mat);
DoubleVector2D matrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2);
void printMlpEnum(MLP::InitMethod inpEnum);
void printMlpEnum(MLP::ActFunc inpEnum);
void printMlpEnum(MLP::LossFunc inpEnum);
void importCSV(std::string_view);


