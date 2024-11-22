#include "mlp.h"
#include "helperfuncs.h"
#include "debug_log.h"

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

// I wrote this function but i have no idea what it does ... why tf are we doing tanh on weights???
/* void Mlp::actFunc(std::vector<std::vector<double>>& inp) { */
/*     for (size_t layer = 0; layer < weights.size(); layer ++) { */
/*         /1* sigmoid(this->weights[layer]); *1/ */
/*         tanh(this->weights[layer]); */
/*     } */
/* } */

// Version 1, might change this to handle biases differently instead of always appending a vector of 1s (this may be slower)
std::vector<std::vector<double>> Mlp::forwardProp(std::vector<std::vector<double>> inpQuery) const {

    DEBUG_LOG("Performing Forward Prop");
    
    // Row of 1s delcared outside of the for loop since row it is consistent for all iterations
    std::vector<double> row1s(inpQuery[0].size(), 1.0);

    // Perform forward prop
    for (int layer = 0; layer < this->weights.size(); layer++){

        // Pre-pad our inpQueryuts with row of 1s
        inpQuery.insert(inpQuery.begin(), row1s);

        // For Debugging
        DEBUG_LOG("`weights` Matrix * `inpQuery` Matrix: " 
                << this->weights[layer].size() << "x" << this->weights[layer][0].size()
                << " * " << inpQuery.size() << "x" << inpQuery[0].size());

        // wieght matrix multipled with our inpQueryut
        inpQuery = matrixMultiply(this->weights[layer], inpQuery);
        // apply activation function
        tanh(inpQuery);
    }
    return inpQuery;
}

void Mlp::backPropIteration(const std::vector<std::vector<double>>& inpBatch) {

    // Forward prop run, then calc neuron differentials then can get weight update with those

    this->forwardProp(std::vector<std::vector<double>> inp)
    std::cout << "HITYA" << std::endl;
}



 

