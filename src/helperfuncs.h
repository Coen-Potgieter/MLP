#include <cstdint>
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

struct DataMNIST {
    Uint8Vector3D imgs;
    std::vector<uint8_t> labels;
};

namespace fs = std::filesystem;

// Printing Functions
void printMlpEnum(MLP::InitMethod inpEnum);
void printMlpEnum(MLP::ActFunc inpEnum);
void printMlpEnum(MLP::LossFunc inpEnum);
void printData(DoubleVector2D data, const size_t& numRows=0);
void printMNISTImg(const Uint8Vector2D& inptImg, const uint8_t& brightnessThreshold=128);
void printMatrix(const DoubleVector2D& mat);
void printDims(const DoubleVector2D& mat);

// Matrix Operations
DoubleVector2D matrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2);
DoubleVector2D elementWiseMatrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2);
DoubleVector2D transpose(const DoubleVector2D& matrix);
double sumMatrixElems(const DoubleVector2D& inpMatrix);
double maxValInVector(const std::vector<double>& inpVec);

// Pre-processing Operations
void initUniformRandom(DoubleVector3D& inp, const double minVal, const double maxVal, const bool isWeights);
void normaliseData(DoubleVector2D& data);
std::vector<std::string> separateRow(const std::string& content);
DoubleVector2D separateTarget(DoubleVector2D& data, std::vector<int>& targetCols);
DoubleVector2D sliceCols(const DoubleVector2D& inpMatrix, const size_t startIdx, const size_t endIdx);
Uint8Vector2D Flatten3DTensor(const Uint8Vector3D& data);
Uint8Vector3D buildImgFromFlat(const Uint8Vector2D& flatData);
DoubleVector2D castImgsFromUint8ToDouble(const Uint8Vector2D& data);
std::vector<double> castTargetsFromUint8ToDouble(const std::vector<uint8_t>& labels);
DoubleVector2D oneHotEncodeTargets(const std::vector<double>& labels);

// Activations And Gradients
double relu(const double z);
double tanh(const double z);
double sigmoid(const double z);
double elu(const double z);
std::vector<double> softmaxHandler(const std::vector<double>& Z, bool normaliseInput=true);
std::vector<double> softmax(const std::vector<double>& Z);

double derivativeRelu(const double z);
double derivativeTanh(const double z);
double derivativeSigmoid(const double z);
double derivativeElu(const double z);
std::vector<double> derivativeSoftmax(const std::vector<double>& Z);

// Importing Functions
DoubleVector2D importCSV(std::string_view pathToCSV);
DataMNIST importMNIST();




