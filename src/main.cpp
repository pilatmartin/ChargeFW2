#include <fmt/format.h>
#include <memory>
#include <filesystem>
#include <cstdio>
#include <sys/resource.h>
#include <ctime>
#include <chrono>
#include <cmath>
#include <unistd.h>
#include <tuple>
#include <algorithm>

#include "chargefw2.h"
#include "formats/reader.h"
#include "formats/mol2.h"
#include "formats/pqr.h"
#include "formats/cif.h"
#include "formats/txt.h"
#include "structures/molecule_set.h"
#include "parameters.h"
#include "charges.h"
#include "method.h"
#include "candidates.h"
#include "config.h"
#include "options.h"
#include "utility/strings.h"
#include "exceptions/file_exception.h"
#include "exceptions/internal_exception.h"

namespace fs = std::filesystem;


int main(int argc, char **argv) {
    auto parsed = parse_args(argc, argv);
    check_common_args();

    auto start = std::chrono::system_clock::now();

    auto ext = std::filesystem::path(config::input_file).extension().string();

    try {
        MoleculeSet m = load_molecule_set(config::input_file);

        if (m.molecules().empty()) {
            fmt::print(stderr, "No molecules were loaded from the input file\n");
            exit(EXIT_FILE_ERROR);
        }

        bool is_protein_structure = m.has_proteins();

        if (config::mode == "info") {
            m.classify_atoms(AtomClassifier::PLAIN);
            m.info();

        } else if (config::mode == "charges") {
            std::string method_name;
            if (config::method_name.empty()) {
                auto methods = get_suitable_methods(m, is_protein_structure, config::permissive_types);
                method_name = std::get<0>(methods.front());
                fmt::print("Autoselecting the best method.\n");
            } else {
                method_name = config::method_name;
            }

            auto method = load_method(method_name);
            fmt::print("Method: {}\n", method->name());

            setup_method_options(method, parsed);

            auto p = std::unique_ptr<Parameters>();

            if (method->has_parameters()) {
                std::string par_name;
                if (config::par_file.empty()) {
                    par_name = best_parameters(m, method, is_protein_structure, config::permissive_types);
                    if (par_name.empty()) {
                        fmt::print(stderr, "No parameters found \n");
                        exit(EXIT_PARAMETER_ERROR);
                    }
                    fmt::print("Best parameters found: {}\n", par_name);
                    par_name = fs::path(INSTALL_DIR) / "share" / "parameters" / par_name;
                } else {
                    par_name = config::par_file;
                }

                try {
                    p = std::make_unique<Parameters>(par_name);
                } catch (std::runtime_error &e) {
                    fmt::print(stderr, "{}\n", e.what());
                    exit(EXIT_FILE_ERROR);
                }

                p->print();

                size_t unclassified = m.classify_set_from_parameters(*p, true, config::permissive_types);
                fmt::print("\nNumber of unclassified molecules: {}\n\n", unclassified);

                try {
                    method->set_parameters(p.get());
                } catch (std::runtime_error &e) {
                    fmt::print(stderr, "{}\n", e.what());
                    exit(EXIT_FILE_ERROR);
                }

            } else {
                m.classify_atoms(AtomClassifier::PLAIN);
            }

            m.info();
            m.fulfill_requirements(method->get_requirements());

            auto charges = Charges();

            charges.set_method_name(method->name());
            if (method->has_parameters()) {
                charges.set_parameters_name(method->parameters()->name());
            }

            for (auto &mol: m.molecules()) {
                auto results = method->calculate_charges(mol);
                if (std::any_of(results.begin(), results.end(), [](double chg) { return not std::isfinite(chg); })) {
                    fmt::print(stderr, "Cannot compute charges for {}: Method returned numerically incorrect values\n",
                            mol.name());
                    continue;
                }
                charges.insert(mol.name(), results);
            }

            std::filesystem::path dir(config::chg_out_dir);
            std::filesystem::path file(config::input_file);
            auto txt_str = file.filename().string() + ".txt";
            TXT().save_charges(m, charges, dir / std::filesystem::path(txt_str));
            CIF().save_charges(m, charges, config::input_file);

            if (is_protein_structure) {
                auto pqr_str = file.filename().string() + ".pqr";
                PQR().save_charges(m, charges, dir / std::filesystem::path(pqr_str));
            } else {
                auto mol2_str = file.filename().string() + ".mol2";
                Mol2().save_charges(m, charges, dir / std::filesystem::path(mol2_str));
            }

            if (not config::log_file.empty()) {
                struct rusage usage = {};
                getrusage(RUSAGE_SELF, &usage);

                double utime = usage.ru_utime.tv_sec + static_cast<double>(usage.ru_utime.tv_usec) / 10e6;
                double stime = usage.ru_stime.tv_sec + static_cast<double>(usage.ru_stime.tv_usec) / 10e6;
                double mem = static_cast<double>(usage.ru_maxrss) / 1024;
                auto now = time(nullptr);
                char current_time[100];
                strftime(current_time, 100, "%c", localtime(&now));
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> walltime = end - start;

                auto pid = getpid();
                auto log_file = std::fopen(config::log_file.c_str(), "a");

                fmt::print(log_file, "{} [{}]; File: {}; Processed molecules: {}; Method: {}; Parameters: {}\n",
                        current_time, pid, config::input_file, m.molecules().size(), method->name(), charges.parameters_name());

                fmt::print(log_file,
                        "{} [{}]; Walltime: {:.2f} s; User time: {:.2f} s; System time: {:.2f} s; Peak memory: {:.1f} MB \n",
                        current_time, pid, walltime.count(), utime, stime, mem);
            }
        } else if (config::mode == "best-parameters") {
            const auto method = load_method(config::method_name);

            if (!method->has_parameters()) {
                fmt::print(stderr, "Method uses no parameters\n");
                exit(EXIT_PARAMETER_ERROR);
            }
            auto best = best_parameters(m, method, is_protein_structure, config::permissive_types);
            if (best.empty()) {
                fmt::print("There are no best parameters\n");
            } else {
                fmt::print("Best parameters are: {}\n", best);
            }
        } else if (config::mode == "suitable-methods") {
            auto methods = get_suitable_methods(m, is_protein_structure, config::permissive_types);
            for (const auto &[method, parameters]: methods) {
                fmt::print("{}", method);
                for (const auto &parameter_set: parameters) {
                    fmt::print(" {}", parameter_set);
                }
                fmt::print("\n");
            }
        }
        else {
            fmt::print(stderr, "Unknown mode {}\n", config::mode);
            exit(EXIT_PARAMETER_ERROR);
        }
    } catch (FileException &e) {
        fmt::print(stderr, "{}\n", e.what());
        exit(EXIT_FILE_ERROR);
    } catch (InternalException &e) {
        fmt::print(stderr, "{}\n", e.what());
        exit(EXIT_INTERNAL_ERROR);
    }

    return EXIT_SUCCESS;
}
