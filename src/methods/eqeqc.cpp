#include <vector>
#include <cmath>
#include <Eigen/LU>
#include <functional>

#include "eqeqc.h"
#include "../parameters.h"
#include "../geometry.h"

CHARGEFW2_METHOD(EQeqC)


Eigen::VectorXd EQeqC::EE_system(const std::vector<const Atom *> &atoms, double total_charge) const {

    const auto n = static_cast<Eigen::Index>(atoms.size());

    const double lambda = 1.2;
    const double k = 14.4;
    double H_electron_affinity = -2.0; // Exception for hydrogen mentioned in the article

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 1, n + 1);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n + 1);
    Eigen::VectorXd J = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd X = Eigen::VectorXd::Zero(n);

    for (Eigen::Index i = 0; i < n; i++) {
        const auto &atom_i = *atoms[i];
        if (atom_i.element().symbol() == "H") {
            X(i) = (atom_i.element().ionization_potential() + H_electron_affinity) / 2;
            J(i) = atom_i.element().ionization_potential() - H_electron_affinity;
        } else {
            X(i) = (atom_i.element().ionization_potential() + atom_i.element().electron_affinity()) / 2;
            J(i) = atom_i.element().ionization_potential() - atom_i.element().electron_affinity();
        }
    }

    for (Eigen::Index i = 0; i < n; i++) {
        const auto &atom_i = *atoms[i];
        A(i, i) = J(i);
        b(i) = -X(i);
        for (Eigen::Index j = i + 1; j < n; j++) {
            const auto &atom_j = *atoms[j];
            double a = std::sqrt(J(i) * J(j)) / k;
            double Rij = distance(atom_i, atom_j);
            double overlap = std::exp(-a * a * Rij * Rij) * (2 * a - a * a * Rij - 1 / Rij);
            auto x = lambda * k / 2 * (1 / Rij + overlap);
            A(i, j) = x;
            A(j, i) = x;
        }
    }

    A.row(n) = Eigen::VectorXd::Constant(n + 1, 1);
    A.col(n) = Eigen::VectorXd::Constant(n + 1, 1);
    A(n, n) = 0;
    b(n) = total_charge;

    return A.partialPivLu().solve(b).head(n);
}


std::vector<double> EQeqC::calculate_charges(const Molecule &molecule) const {
    const auto n = static_cast<Eigen::Index>(molecule.atoms().size());

    auto f = [this](const std::vector<const Atom *> &atoms, double total_charge) -> Eigen::VectorXd {
        return EE_system(atoms, total_charge);
    };

    Eigen::VectorXd q = solve_EE(molecule, f);

    for (Eigen::Index i = 0; i < n; i++) {
        const auto &atom_i = molecule.atoms()[i];
        double correction = 0;
        for (Eigen::Index j = 0; j < n; j++) {
            if (i == j)
                continue;
            const auto &atom_j = molecule.atoms()[j];
            double tkk = parameters_->atom()->parameter(atom::Dz)(atom_i) - parameters_->atom()->parameter(atom::Dz)(atom_j);
            double bkk = std::exp(-parameters_->common()->parameter(common::alpha) *
                                  (distance(atom_i, atom_j) - atom_i.element().covalent_radius() -
                                   atom_j.element().covalent_radius()));
            correction += tkk * bkk;
        }
        q(i) += correction;
    }
    return {q.data(), q.data() + q.size()};
}
