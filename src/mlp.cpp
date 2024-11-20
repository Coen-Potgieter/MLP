#include "mlp.h"

Mlp::Mlp() {
    this->structure[] = { 2, 5, 1};
}
Mlp::Mlp(int* inpStructure) {
    this->structure = inpStructure;
}
