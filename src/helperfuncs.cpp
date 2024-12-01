

#include "helperfuncs.h"
#include "mlp.h"
#include <filesystem>
#include <stdexcept>
#include <string>

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
        case MLP::LossFunc::MSE:
            std::cout << "MSE" << std::endl;
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

// Removes target from data and returns it 
// Assumpution: Target col is last column in `data` matrix
DoubleVector2D separateTarget(DoubleVector2D& data) {

    const size_t targetCol = data[0].size() - 1;
    const size_t numRows = data.size();
    DoubleVector2D target(numRows, -1);

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++ ) {
        target[rowIdx] = data[rowIdx][targetCol];
        data[rowIdx].erase(data[rowIdx].begin() + targetCol);
    }

    return target;
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

DoubleVector2D sliceRows(const DoubleVector2D& inpMatrix, const size_t startIdx, const size_t endIdx) {

    // Check for valid Idx Ranges
    if ((startIdx > endIdx) || (endIdx >= inpMatrix.size())) {
        throw std::invalid_argument("Error: Invalid Index Values Given");
    }

    DoubleVector2D outp(endIdx - startIdx + 1);


    for (size_t idx = 0; idx <= endIdx; idx++) {
        outp[idx] = inpMatrix[startIdx + idx];
    }

    return outp;

}




 

