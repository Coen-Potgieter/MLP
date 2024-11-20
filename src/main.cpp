#include "mlp.h"

int main() {
    std::vector<int> myStruct = { 2, 10, 5};
    Mlp myMLP(myStruct);

    myMLP.printWeights();
}
