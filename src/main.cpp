#include "mlp.h"
#include "helperfuncs.h"

int main() {


    
    /* std::vector<std::vector<double>> myInput = { {1,2,3}, {4,5,6}, {7,8,9}, {10,11,12} }; */
    std::vector<std::vector<double>> myInput2 = { {1,2,3,4}, {4,5,6, 7}, {7,8,9, 10}, {3,4,6,1}};
    std::vector<std::vector<double>> myInput = { {1,2,3,5}, {4,5,6,7}, {7,8,9,10} };


    /* std::vector<double> row1s(myInput[0].size(), 1.0); */
    /* myInput.insert(myInput.begin(), row1s); */
    /* std::cout << myInput[0].size() << std::endl; */
    /* /1* printMatrix(myInput); *1/ */
    /* /1* printMatrix(myInput2); *1/ */
    /* std::vector<std::vector<double>> idk = matrixMultiply(myInput2, myInput); */
    /* printMatrix(idk); */
    /* return 0; */

    std::vector<int> myStruct = { 3, 10, 5};
    Mlp myMLP(myStruct);
    myMLP.initWeights(Mlp::InitMethod::UNIFORM);
    myMLP.initBias(Mlp::InitMethod::UNIFORM, -10, 10);
    /* myMLP.printWeights(); */
    /* return 0; */
    std::vector<std::vector<double>> outp = myMLP.forwardProp(myInput);
    printMatrix(outp);
}
