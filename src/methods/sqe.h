#pragma once

#include <vector>

#include "../structures/molecule.h"
#include "../method.h"


class SQE : public Method {
    enum atom{electronegativity, hardness, width};
    enum bond{kappa};

public:
    explicit SQE() : Method("SQE", {}, {"electronegativity", "hardness", "width"}, {"kappa"}, {}) {}

    [[nodiscard]] const MethodMetadata& get_metadata() const override;
    
    [[nodiscard]] std::vector<double> calculate_charges(const Molecule &molecule) const override;

    [[nodiscard]] bool is_suitable_for_large_molecule() const override { return false; }
};
