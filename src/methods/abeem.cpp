#include <vector>
#include <cmath>
#include <Eigen/LU>

#include "abeem.h"
#include "../parameters.h"
#include "../geometry.h"

CHARGEFW2_METHOD(ABEEM)

std::vector<double> ABEEM::calculate_charges(const Molecule &molecule) const {

    const auto n = static_cast<Eigen::Index>(molecule.atoms().size());
    const auto m = static_cast<Eigen::Index>(molecule.bonds().size());
    const auto mn = n + m + 1;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(mn, mn);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(mn);

    const double k = parameters_->common()->parameter(common::k);

    // atom-atom part
    for (Eigen::Index i = 0; i < n; i++) {
        const auto &atom_i = molecule.atoms()[i];
        A(i, i) = parameters_->atom()->parameter(atom::b)(atom_i);
        b(i) = -parameters_->atom()->parameter(atom::a)(atom_i);
        for (Eigen::Index j = i + 1; j < n; j++) {
            const auto &atom_j = molecule.atoms()[j];
            double off = k / distance(atom_i, atom_j);
            A(i, j) = off;
            A(j, i) = off;
        }
    }

    // atom-bond part
    for (Eigen::Index i = 0; i < n; i++) {
        const auto &atom = molecule.atoms()[i];
        for (Eigen::Index j = 0; j < m; j++) {
            const auto &bond = molecule.bonds()[j];
            if (bond.hasAtom(atom)) {
                A(i, n + j) = parameters_->atom()->parameter(atom::c)(atom);
            } else {
                A(i, n + j) = k / distance(atom, bond, true);
            }
        }

    }

    // bond-atom part
    for (Eigen::Index i = 0; i < m; i++) {
        const auto &bond = molecule.bonds()[i];
        b(n + i) = -parameters_->bond()->parameter(bond::A)(bond);
        for (Eigen::Index j = 0; j < n; j++) {
            const auto &atom = molecule.atoms()[j];
            if (bond.hasAtom(atom)) {
                if (bond.first() == atom) {
                    A(n + i, j) = parameters_->bond()->parameter(bond::D)(bond);
                } else {
                    A(n + i, j) = parameters_->bond()->parameter(bond::C)(bond);
                }
            } else {
                A(n + i, j) = k / distance(atom, bond, true);
            }
        }
    }

    // bond-bond part
    for (Eigen::Index i = 0; i < m; i++) {
        const auto &bond_i = molecule.bonds()[i];
        A(n + i, n + i) = parameters_->bond()->parameter(bond::B)(bond_i);
        for (Eigen::Index j = i + 1; j < m; j++) {
            const auto &bond_j = molecule.bonds()[j];
            double off = k / distance(bond_i, bond_j, true);
            A(n + i, n + j) = off;
            A(n + j, n + i) = off;
        }
    }

    for (Eigen::Index i = 0; i < n + m; i++) {
        A(i, n + m) = 1;
        A(n + m, i) = 1;
    }

    A(n + m, n + m) = 0;
    b(n + m) = molecule.total_charge();

    Eigen::VectorXd q = A.partialPivLu().solve(b).head(mn);

    // Redistribute the bond charges to the corresponding atoms
    for(Eigen::Index i = 0; i < m; i++) {
        const auto &bond = molecule.bonds()[i];
        q(static_cast<Eigen::Index>(bond.first().index())) += 0.5 * q(n + i);
        q(static_cast<Eigen::Index>(bond.second().index())) += 0.5 * q(n + i);
    }

    return {q.data(), q.data() + n};
}
