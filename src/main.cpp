#include "mlp.h"
#include "helperfuncs.h"

int main() {


    /* std::vector<std::vector<double>> myInput = { {1,2,3}, {4,5,6}, {7,8,9}, {10,11,12} }; */
    std::vector<std::vector<double>> myInput = { {1,2,3}, {4,5,6}, {7,8,9} };
    std::vector<std::vector<double>> myInput2 = { {1,2,3,5}, {4,5,6,7}, {7,8,9,10} };

    try{
        std::vector<std::vector<double>> outp = matrixMultiply(myInput, myInput2);
        printMatrix(outp);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
    std::vector<int> myStruct = { 2, 10, 5};
    Mlp myMLP(myStruct);
    myMLP.initWeights(Mlp::InitMethod::UNIFORM);
    myMLP.initBias(Mlp::InitMethod::UNIFORM, -10, 10);
    /* myMLP.actFunc(); */
    myMLP.printWeights();
}
