#include <Eigen/LU>

#include "delre.h"
#include "../structures/molecule.h"
#include "../structures/bond.h"
#include "../parameters.h"

CHARGEFW2_METHOD(DelRe)


std::vector<double> DelRe::calculate_charges(const Molecule &molecule) const {

    const auto n = static_cast<Eigen::Index>(molecule.atoms().size());
    const auto m = static_cast<Eigen::Index>(molecule.bonds().size());

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

    for (Eigen::Index i = 0; i < n; i++) {
        auto &atom_i = molecule.atoms()[i];
        b(i) = -parameters_->atom()->parameter(atom::delta)(atom_i);
        A(i, i) = -1.0;
    }

    for (const auto &bond: molecule.bonds()) {
        auto i = static_cast<Eigen::Index>(bond.first().index());
        auto j = static_cast<Eigen::Index>(bond.second().index());
        A(i, j) = parameters_->bond()->parameter(bond::gammaA)(bond);
        A(j, i) = parameters_->bond()->parameter(bond::gammaB)(bond);
    }

    Eigen::VectorXd d = A.partialPivLu().solve(b);
    std::vector<double> q(n, 0);

    for (Eigen::Index k = 0; k < m; k++) {
        const auto &bond = molecule.bonds()[k];
        auto i = static_cast<Eigen::Index>(bond.first().index());
        auto j = static_cast<Eigen::Index>(bond.second().index());
        double dq = (d(i) - d(j)) / (2 * parameters_->bond()->parameter(bond::eps)(bond));
        q[i] -= dq;
        q[j] += dq;
    }

    return q;
}
