#include "mlp.h"

Mlp::Mlp(const std::vector<int>& inpStructure){

    // Loop through each Layer
    for (size_t layerIdx = 0; layerIdx < inpStructure.size() - 1; layerIdx++) {
        const int cols = inpStructure[layerIdx];
        const int rows = inpStructure[layerIdx + 1];

        // Construct the weight matrix
        std::vector<std::vector<double>> weightMatrix(rows, std::vector<double>(cols, 0.0));

        for (size_t row = 0; row < inpStructure[layerIdx+1]; row++) {
            for (size_t col = 0; col<inpStructure[layerIdx]; col++) {
                double randWeight = (rand()% 2001 - 1000) / 1000.0;
                weightMatrix[row][col] = randWeight;
            }

        }

        // Now assign wieght matrix to out member variable
        this->weights.push_back(weightMatrix);
    }
}

Mlp::~Mlp() = default;

void Mlp::printWeights() const {
    for (size_t layer = 0; layer < weights.size(); layer ++) {
        for (size_t row = 0; row < weights[layer].size(); row++){
            for (size_t col = 0; col < weights[layer][row].size(); col++){
                std::cout << weights[layer][row][col] << ", ";
            }
        }
    }
    std::cout << std::endl;
}





 

