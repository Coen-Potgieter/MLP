#include "helperfuncs.h"
#include "debug_log.h"
#include "mlp.h"
#include <cmath>
#include <cstdlib>

void printMatrix(const DoubleVector2D& mat) {
    for (const std::vector<double>& row : mat) {
        for (const double& elem : row) {
            // Adjust the width for better alignment
            std::cout << " | " << elem;
        }
        std::cout << std::endl; // New line after each row
    }
}
void printDims(const DoubleVector2D& mat) {
    std::cout << "Dimensions: (" << mat.size() << "x" << mat[0].size() << ")" << std::endl;
}

// Activations And Gradients
double relu(const double z) {
    if (z < 0) {
        return 0;
    } else {
        return z;
    }
}
double tanh(const double z) { 
    return (std::exp(z) - std::exp(-z)) / (std::exp(z) + std::exp(-z));
}
double sigmoid(const double z) {
    return 1 / (1 + std::exp(-z)); 
}
double elu(const double z) {
    if (z > 0) {
        return z;
    } else {
        return std::exp(z) - 1;
    }
}


std::vector<double> softmax(const std::vector<double>& Z) {
    std::vector<double> expVals;
    double expSum = 0.00001;
    for (const double& z : Z) {
        double expZ = std::exp(z);
        expVals.push_back(expZ);
        expSum += expZ;
    }

    for (size_t i = 0; i < expVals.size(); i++) {
        expVals[i] /= expSum;
    }
    return expVals;
}
double derivativeRelu(const double z) {
    if (z < 0) {
        return 0;
    } else {
        return 1;
    }
}
double derivativeTanh(const double z) {
    const double intermediateVal = tanh(z);
    return 1 - intermediateVal * intermediateVal;
}
double derivativeSigmoid(const double z) {
    const double intermediateVal = sigmoid(z);
    return intermediateVal * ( 1- intermediateVal);
}
double derivativeElu(const double z) {
    if (z > 0){
        return 1;
    } else {
        return std::exp(z);
    }
}

DoubleVector2D matrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2) {

    // Ensure that dimensions are correct (mat1 cols == mat2 rows)
    if (mat1[0].size() != mat2.size()){
        DEBUG_LOG("Dims of `mat1`: " << mat1.size() << "x" << mat1[0].size() <<
                  " | Dims of `mat2`: " << mat2.size() << "x" << mat2[0].size());
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

DoubleVector2D elementWiseMatrixMultiply(const DoubleVector2D& mat1, const DoubleVector2D& mat2) {

    // Ensure they are the same size
    if ((mat1.size() != mat2.size()) || (mat1[0].size() != mat2[0].size())) {
        throw std::invalid_argument("`mat1` is not of same size as `mat2`");
    }

    // Create output matrix
    const size_t numRows = mat1.size();
    const size_t numCols = mat1[0].size();
    DoubleVector2D outp(numRows, std::vector<double>(numCols));

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
        for (size_t colIdx = 0; colIdx < numCols; colIdx++) {
            outp[rowIdx][colIdx] = mat1[rowIdx][colIdx] * mat2[rowIdx][colIdx];
        }
    }
    return outp;
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
void printMlpEnum(MLP::ActFunc inpEnum) {

    switch(inpEnum) {
        case MLP::ActFunc::SIGMOID:
            std::cout << "SIGMOID" << std::endl;
            break;
        case MLP::ActFunc::TANH:
            std::cout << "TANH" << std::endl;
            break;
        case MLP::ActFunc::RELU:
            std::cout << "RELU" << std::endl;
            break;
        case MLP::ActFunc::ELU:
            std::cout << "ELU" << std::endl;
            break;
        default:
            std::cout << "Printing for this not implemented just yet, so get here and do it" << std::endl;
    }
    return;
}
void printMlpEnum(MLP::LossFunc inpEnum) {

    switch(inpEnum) {
        case MLP::LossFunc::SSE:
            std::cout << "SSE" << std::endl;
            break;
        case MLP::LossFunc::ENTROPY:
            std::cout << "ENTROPY" << std::endl;
            break;
        default:
            std::cout << "Printing for this not implemented just yet, so get here and do it" << std::endl;
    }
}

std::vector<std::string> separateRow(const std::string& content) {
    std::vector<std::string> result;
    size_t start=0, end=0;
    while ((end = content.find(',', start)) != std::string::npos) {
        result.push_back(content.substr(start, end-start));
        start = end + 1;
    }
    result.push_back(content.substr(start));
    return result;
}


DoubleVector2D importCSV(std::string_view pathToCSV) {

    DoubleVector2D outp;

    // Check if input is a valid path to a csv
    fs::path fp = pathToCSV;
    if (fp.extension() != ".csv" || !fs::exists(fp)) {
        throw std::invalid_argument("Error: Given File Path Does Not Point to a .csv File");
    }

    // Open File and throw error if fails
    std::ifstream fin;
    fin.open(pathToCSV);
    if (!fin) {
        throw std::ios_base::failure("Error: Could Not Open The File");
    }

    std::string line;
    std::vector<std::string> row;

    std::string possibleDegrees[] = { "bachelor", "highschool" };

    std::getline(fin, line);
    while (std::getline(fin, line)) {

        std::vector<double> procRow(6, 0);

        row = separateRow(line);
        // Process each element

        // Populate Degree 
        if (row[2] == "bachelor") {
            procRow[0] = 1;
        } else if (row[2] == "highschool") {
            procRow[1] = 1;
        } else {
            std::cerr << "Degree: " << row[2] << " Is Not Accounted For" << std::endl;
        }

        // Populate Gender
        if (row[3] == "male") {
            procRow[2] = 1;
        } else if (row[3] == "female") {
            procRow[3] = 1;
        } else {
            std::cerr << "Gender: " << row[3] << " Is Not Accounted For" << std::endl;
        }

        // Populate Age
        procRow[4] = std::stod(row[4]);

        // Populate Earnings
        procRow[5] = std::stod(row[1]);

        outp.push_back(procRow);
    }

    fin.close();
    return outp;
}

// z-Score Normalisation
void normaliseData(DoubleVector2D& data) {

    // Get mean
    double runningSum = 0;
    const size_t numRows = data.size();
    for (const std::vector<double>& row : data) {
        runningSum += row[4];
    }
    double mean = runningSum / numRows;
    /* std::cout << mean << std::endl; */

    // Get Standard Dev
    runningSum = 0;
    for (const std::vector<double>& row : data) {
        runningSum += (row[4] - mean) * (row[4] - mean);
    }
    double sd = std::sqrt(runningSum / (numRows - 1));

    // Update each age value with mean and sd (I back i should avoid range based for loop? for clairty)
    for (std::vector<double>& row : data) {
        row[4] = (row[4] - mean) / sd;
    }

    return;
}

// Removes target from data and returns it as a matrix
DoubleVector2D separateTarget(DoubleVector2D& data, std::vector<int>& targetCols) {

    // Ensure targetCols are not empty
    if (targetCols.size() <= 0) {
        throw std::invalid_argument("List of target columns is empty");
    }
    // Ensure indicies are inside `targetCols` is within `data` columns
    for (const int& targetIndex : targetCols) {
        if ((targetIndex < 0) || (targetIndex >= data[0].size())){
            throw std::invalid_argument("Indicies inside `targetCols` are out of range");
        }
    }

    const size_t numRows = data.size();
    const size_t numTargets = targetCols.size();
    DoubleVector2D targetOutput(numRows, std::vector<double>(numTargets));

    // I need to sort the vector using this implementation
    std::sort(targetCols.begin(), targetCols.end(), std::greater<int>());

    size_t targetIdx = 0;
    for (size_t targetCol : targetCols) {
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++ ) {
            targetOutput[rowIdx][targetIdx] = data[rowIdx][targetCol];
            data[rowIdx].erase(data[rowIdx].begin() + targetCol);
        }
        targetIdx += 1;
    }
    return targetOutput;
}

void printData(DoubleVector2D data, const size_t& numRows) {

    size_t rowIdx = 1;
    for (const std::vector<double>& row : data) {

        std::cout << "Row " << rowIdx << std::endl;
        for (const double& elem : row) {
            std::cout << elem << " | ";
        }
        std::cout << std::endl << std::endl;

        if (rowIdx == numRows) {
            break;
        }
        rowIdx += 1;
    }
    return;
}

// Taken From ChatGPT
DoubleVector2D transpose(const DoubleVector2D& matrix) {
    if (matrix.empty()) return {}; // Handle empty matrix

    size_t numRows = matrix.size();
    size_t numCols = matrix[0].size();

    // Initialize a transposed matrix with swapped dimensions
    std::vector<std::vector<double>> transposed(numCols, std::vector<double>(numRows));

    // Fill the transposed matrix
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

DoubleVector2D sliceCols(const DoubleVector2D& inpMatrix, const size_t startIdx, const size_t endIdx) {

    // Check for valid Idx Ranges
    if ((startIdx > endIdx) || (endIdx >= inpMatrix[0].size())) {
        throw std::invalid_argument("Error: Invalid Index Values Given");
    }
    const size_t numRows = inpMatrix.size();
    const size_t numCols = endIdx - startIdx + 1;
    DoubleVector2D outp(numRows, std::vector<double>(numCols));

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
        for (size_t colIdx = 0; colIdx < numCols; colIdx++) {
            outp[rowIdx][colIdx] = inpMatrix[rowIdx][startIdx + colIdx];
        }
    }
    return outp;
}
double sumMatrixElems(const DoubleVector2D& inpMatrix) {
    double runningSum = 0;
    for (const std::vector<double>& row : inpMatrix) {
        for (const double& elem : row) {
            runningSum += elem;
        }
    }
    return runningSum;
}


DataMNIST importMNIST() {
    // Define path
    std::string_view pathToImgs = "data/t10k-images.idx3-ubyte";

    DEBUG_LOG("\nImporting MNIST dataset:\nImage Data From: " << pathToImgs);
    std::ifstream fin;
    fin.open(pathToImgs, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to Open File At: " << pathToImgs << std::endl;
    }

    // Read headers
    int32_t magicNumber = 0;
    int32_t numImgs = 0;
    int32_t numRows = 0;
    int32_t numCols = 0;

    fin.read(reinterpret_cast<char*>(&magicNumber), 4);
    if (fin.gcount() < 4) {
        std::cerr << "Error Reading Magic Number" << std::endl;
    }
    fin.read(reinterpret_cast<char*>(&numImgs), 4);
    if (fin.gcount() < 4) {
        std::cerr << "Error Reading Number of Images" << std::endl;
    }
    fin.read(reinterpret_cast<char*>(&numRows), 4);
    if (fin.gcount() < 4) {
        std::cerr << "Error Reading Number of Rows" << std::endl;
    }
    fin.read(reinterpret_cast<char*>(&numCols), 4);
    if (fin.gcount() < 4) {
        std::cerr << "Error Reading Number of Cols" << std::endl;
    }

    // Got Fucky results, so I'm reversing endianess
    magicNumber = __builtin_bswap32(magicNumber);
    numImgs = __builtin_bswap32(numImgs);
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    DEBUG_LOG("Magic Number: " << magicNumber << std::endl << 
              "Number of Images: " << numImgs << std::endl << 
              "Number of Rows: " << numRows << std::endl << 
              "Number of Cols: " << numCols << std::endl 
              );

    std::vector<std::vector<std::vector<uint8_t>>> imgs(
        numImgs, 
        std::vector<std::vector<uint8_t>>(
            numRows, std::vector<uint8_t>(
                numCols))
    );

    // Read image data
    for (int i = 0; i < numImgs; ++i) {
        for (int r = 0; r < numRows; ++r) {
            fin.read(reinterpret_cast<char*>(imgs[i][r].data()), numCols);
            if (fin.gcount() < 1) {
                std::cerr << "Error Reading Image: " << i << " and Row: " << r << std::endl;
            }
        }
    }
    fin.close();


    // Define path
    std::string_view pathToLabels = "data/t10k-labels.idx1-ubyte";


    DEBUG_LOG("Import Label Data From: " << pathToLabels);
    std::ifstream finLabels;
    finLabels.open(pathToLabels, std::ios::binary);
    if (!finLabels) {
        std::cerr << "Failed to Open File At: " << pathToLabels << std::endl;
    }

    // Read headers
    magicNumber = 0;
    int32_t numLabels = 0;

    finLabels.read(reinterpret_cast<char*>(&magicNumber), 4);
    if (finLabels.gcount() < 4) {
        std::cerr << "Error Reading Magic Number" << std::endl;
    }
    finLabels.read(reinterpret_cast<char*>(&numLabels), 4);
    if (finLabels.gcount() < 4) {
        std::cerr << "Error Reading Number of Labels" << std::endl;
    }

    // Got Fucky results, so I'm reversing endianess
    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    DEBUG_LOG("Magic Number: " << magicNumber << std::endl <<
              "Number of Labels: " << numLabels << std::endl);

    std::vector<uint8_t> labels(numLabels);
    // Read label data
    finLabels.read(reinterpret_cast<char*>(labels.data()), numLabels);
    if (finLabels.gcount() < numLabels) {
        std::cerr << "Error Reading Labels" << std::endl;
    }
    finLabels.close();

    DataMNIST results;
    results.imgs = imgs;
    results.labels = labels;
    return results;
}

void printMNISTImg(const Uint8Vector2D& inptImg, const uint8_t& brightnessThreshold) {

    size_t numRows = inptImg.size();
    size_t numCols = inptImg[0].size();

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
        for (size_t colIdx = 0; colIdx < numCols; colIdx++){
            uint8_t pixelVal = inptImg[rowIdx][colIdx];
            std::string_view pixelVisual = (pixelVal >= brightnessThreshold) ? "#" : "-";
            std::cout << pixelVisual;
        }
        std::cout << std::endl;
    }
}

Uint8Vector2D Flatten3DTensor(const Uint8Vector3D& data) {

    const int numImgs = data.size();
    const int numRows = data[0].size();
    const int numCols = data[0][0].size();

    Uint8Vector2D outp(numRows*numCols, std::vector<uint8_t>(numImgs));

    int targetFlatRowIdx;
    for (size_t imgIdx = 0; imgIdx < numImgs; imgIdx++){
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++){
            for (size_t colIdx = 0; colIdx < numCols; colIdx++){
                targetFlatRowIdx = numCols*rowIdx + colIdx;
                outp[targetFlatRowIdx][imgIdx] = data[imgIdx][rowIdx][colIdx];
            }
        }
    }

    return outp;
}

Uint8Vector3D buildImgFromFlat(const Uint8Vector2D& flatData) {

    const int numImgs = flatData[0].size();
    const int numRows = sqrt(flatData.size());
    const int numCols = numRows;

    Uint8Vector3D outp(
        numImgs, 
        std::vector<std::vector<uint8_t>>(
            numRows, std::vector<uint8_t>(
                numCols))
    );

    int targetFlatRowIdx;
    for (size_t imgIdx = 0; imgIdx < numImgs; imgIdx++){
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++){
            for (size_t colIdx = 0; colIdx < numCols; colIdx++){
                targetFlatRowIdx = numCols*rowIdx + colIdx;
                outp[imgIdx][rowIdx][colIdx] = flatData[targetFlatRowIdx][imgIdx];
            }
        }
    }

    return outp;
}

DoubleVector2D castImgsFromUint8ToDouble(const Uint8Vector2D& data) {
    const int numRows = data.size();
    const int numCols = data[0].size();

    DoubleVector2D outp(numRows, std::vector<double>(numCols));

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++){
        for (size_t colIdx = 0; colIdx < numCols; colIdx++){
            outp[rowIdx][colIdx] = static_cast<double>(data[rowIdx][colIdx]);
        }
    }
    return outp;
}


std::vector<double> castTargetsFromUint8ToDouble(const std::vector<uint8_t>& labels) {
    const int numLabels = labels.size();
    std::vector<double> outp(numLabels);
    for (int l = 0; l < numLabels; l++) {
        outp[l] = static_cast<double>(labels[l]);
    }
    return outp;
}


DoubleVector2D oneHotEncodeTargets(const std::vector<double>& labels) {

    const int numLabels = labels.size();

    DoubleVector2D outp(10, std::vector<double>(numLabels, 0.0));

    uint8_t oneHotLabelIdx;
    for (size_t labelIdx = 0; labelIdx < numLabels; labelIdx++) {
        oneHotLabelIdx = labels[labelIdx];
        outp[oneHotLabelIdx][labelIdx] = 1.0;
    }
    return outp;
}








