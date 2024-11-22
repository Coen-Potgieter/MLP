#include "mlp.h"
#include "helperfuncs.h"

int main() {
    std::vector<std::vector<double>> myInput = { {1,2,3,5}, {4,5,6,7}, {7,8,9,10} };

    std::vector<int> myStruct = { 3, 10, 5};
    Mlp myMLP(myStruct);
    myMLP.initWeights(Mlp::InitMethod::UNIFORM);
    myMLP.initBias(Mlp::InitMethod::UNIFORM, -10, 10);
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> res = myMLP.forwardProp(myInput);

    std::cout << std::endl;
    printMatrix(res.second[1]);
}
