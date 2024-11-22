#include "mlp.h"
#include "helperfuncs.h"

void testGettersSetters();
int main() {
    std::vector<int> myStruct = { 3, 10, 5};
    MLP mlp(myStruct);

    std::vector<double> groundTruth = { 3.4, 1.1 };
    std::vector<double> results = { 3.9, 2.1 };
    std::vector<double> errors = mlp.calcLoss(groundTruth, results);

    for (const double& elem : errors) {
        std::cout << elem << std::endl;
    }
    return 0;
    DoubleVector2D myInput = { {1,2,3,5}, {4,5,6,7}, {7,8,9,10} };

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

    std::cout << std::endl;
    std::cout << "Hidden Layer Activation Testing" << std::endl;
    printMlpEnum(mlp.getHiddenLayerAct());
    mlp.setHiddenLayerAct(MLP::ActFunc::ELU);
    printMlpEnum(mlp.getHiddenLayerAct());

    // Output layer act 
    std::cout << std::endl;
    std::cout << "Output Layer Activation Testing" << std::endl;
    printMlpEnum(mlp.getOutputLayerAct());
    mlp.setOutputLayerAct(MLP::ActFunc::ELU);
    printMlpEnum(mlp.getOutputLayerAct());
    
    // loss func 
    std::cout << std::endl;
    std::cout << "Loss Func Testing" << std::endl;
    printMlpEnum(mlp.getLossFunc());
    mlp.setLossFunc(MLP::LossFunc::ENTROPY);
    printMlpEnum(mlp.getLossFunc());

    // LR 
    std::cout << std::endl;
    std::cout << "Learning Rate Testing" << std::endl;
    std::cout << mlp.getLR() << std::endl;
    mlp.setLR(0.4);
    std::cout << mlp.getLR() << std::endl;

    // DecayRate 
    std::cout << std::endl;
    std::cout << "Decay Rate Testing" << std::endl;
    std::cout << mlp.getDecayRate() << std::endl;
    mlp.setDecayRate(90);
    std::cout << mlp.getDecayRate() << std::endl;
    
    // batchSize 
    std::cout << std::endl;
    std::cout << "Batch Size Testing" << std::endl;
    std::cout << mlp.getBatchSize() << std::endl;
    mlp.setBatchSize(10);
    std::cout << mlp.getBatchSize() << std::endl;
}
