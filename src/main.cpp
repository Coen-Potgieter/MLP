#include "mlp.h"
#include "helperfuncs.h"

int main() {
    DoubleVector2D myInput = { {1,2,3,5}, {4,5,6,7}, {7,8,9,10} };

    std::vector<int> myStruct = { 3, 10, 5};
    MLP mlp(myStruct);
    /* MLP mlp(myStruct); */
    mlp.initWeights(MLP::InitMethod::UNIFORM);
    mlp.initBias(MLP::InitMethod::UNIFORM, -10, 10);

    // test constructor 

/*     std::cout << mlp.getLR() << std::endl; */
     

    // test getters and setters
    if (mlp.getLossFunc() == MLP::LossFunc::MSE) {
        std::cout << "MSE" << std::endl;
    } else if (mlp.getLossFunc() == MLP::LossFunc::ENTROPY) {
        std::cout << "ENTROPY" << std::endl;
    }

    mlp.setLossFunc(MLP::LossFunc::ENTROPY);
    if (mlp.getLossFunc() == MLP::LossFunc::MSE) {
        std::cout << "MSE" << std::endl;
    } else if (mlp.getLossFunc() == MLP::LossFunc::ENTROPY) {
        std::cout << "ENTROPY" << std::endl;
    }
    /* std::cout << mlp.getLR() << std::endl; */

    ForwardPropResult res = mlp.forwardProp(myInput);

}

void testGettersSetters() {
    std::vector<int> myStruct = { 3, 10, 5};
    MLP mlp(myStruct);

    // Weights: TODO
    //
    // hidden layer act 

    if (mlp.getHiddenLayerAct() == MLP::ActFunc::SIGMOID) {
        std::cout << "SIGMOID" << std::endl;
    } else if (mlp.getHiddenLayerAct() == MLP::ActFunc::TANH) {
        std::cout << "TANH" << std::endl;
    } else {
        std::cout << "ELSE" << std::endl;
    }
    mlp.setHiddenLayerAct(MLP::ActFunc::TANH);
    // output layer act 
    // loss func 
    // LR 
    // DecayRate 
    // batchSize 
    if (mlp.getLossFunc() == MLP::LossFunc::MSE) {
        std::cout << "MSE" << std::endl;
    } else if (mlp.getLossFunc() == MLP::LossFunc::ENTROPY) {
        std::cout << "ENTROPY" << std::endl;
    }

    mlp.setLossFunc(MLP::LossFunc::ENTROPY);
    if (mlp.getLossFunc() == MLP::LossFunc::MSE) {
        std::cout << "MSE" << std::endl;
    } else if (mlp.getLossFunc() == MLP::LossFunc::ENTROPY) {
        std::cout << "ENTROPY" << std::endl;
    }
    /* std::cout << mlp.getLR() << std::endl; */
}
