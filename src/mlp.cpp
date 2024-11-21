#include "mlp.h"
#include "helperfuncs.h"

Mlp::Mlp(const std::vector<int>& inpStructure){

    // Loop through each Layer
    for (size_t layerIdx = 0; layerIdx < inpStructure.size() - 1; layerIdx++) {
        const int cols = inpStructure[layerIdx];
        const int rows = inpStructure[layerIdx + 1];

        // Construct the empty weight matrix (cols plus 1 for each bias)
        std::vector<std::vector<double>> weightMatrix(rows, std::vector<double>(cols + 1, 0.0));

        // Now assign wieght matrix to out member variable
        this->weights.push_back(weightMatrix);
    }
}

Mlp::~Mlp() = default;

void Mlp::printWeights() const {
    std::cout << std::endl;
    for (size_t layer = 0; layer < weights.size(); layer ++) {
        std::cout << "Pinting Layer: " << layer << std::endl;
        printMatrix(this->weights[layer]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}
void Mlp::initWeights(InitMethod method, const int minVal, const int maxVal) {

    for (size_t layer = 0; layer < weights.size(); layer ++) {
        for (size_t row = 0; row < weights[layer].size(); row++){
            // Note the start index of col here since the first column is reserved for bias
            for (size_t col = 1; col < weights[layer][row].size(); col++){
                const double randomWeight = (rand() % ((maxVal - minVal) * 1000)) / 1000.0 + minVal;
                this->weights[layer][row][col] = randomWeight;
            }
        }
    }
}

void Mlp::initBias(InitMethod method, const int minVal, const int maxVal) {


    for (size_t layer = 0; layer < this->weights.size(); layer ++) {
        for (size_t row = 0; row < this->weights[layer].size(); row++){
            // Uniform
            const double randomWeight = (rand() % ((maxVal - minVal) * 1000)) / 1000.0 + minVal;
            // Only populate the first column
            this->weights[layer][row][0] = randomWeight;
        }
    }
}

void Mlp::actFunc() {
    for (size_t layer = 0; layer < weights.size(); layer ++) {
        /* sigmoid(this->weights[layer]); */
        tanh(this->weights[layer]);
    }
}
std::vector<std::vector<double>> Mlp::forwardProp(std::vector<std::vector<double>>& inp) const {
    
    // Insert row of 1s
    std::vector<double> row1s(inp[0].size(), 1.0);
    inp.insert(inp.begin(), row1s);

    // Perform forward prop
    for (int layer = 0; layer < this->weights.size(); layer++){

        
    }
}




 

