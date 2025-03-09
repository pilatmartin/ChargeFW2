#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <vector>

#include "formats/cif.h"
#include "formats/mol2.h"
#include "formats/pqr.h"
#include "formats/txt.h"
#include "structures/molecule_set.h"
#include "formats/reader.h"
#include "config.h"
#include "candidates.h"
#include "utility/strings.h"
#include "exceptions/file_exception.h"


namespace fs = std::filesystem;
namespace py = pybind11;
using namespace pybind11::literals;

std::map<std::string, std::vector<double>>
calculate_charges(struct Molecules &molecules, const std::string &method_name, std::optional<const std::string> &parameters_name, std::optional<const std::string> &chg_out_dir);

std::vector<py::dict> get_available_methods();

std::vector<std::string> get_available_parameters(const std::string &method_name);

std::vector<std::tuple<std::string, std::vector<std::string>>> get_sutaible_methods_python(struct Molecules &molecules);

py::dict get_parameters_metadata(const std::string &parameters_name);

py::dict atom_type_count_to_dict(const MoleculeSetStats::AtomTypeCount &atom_type_count);

py::dict molecule_info_to_dict(const MoleculeSetStats &info);

void save_charges(const Molecules &molecules, const Charges &charges, const std::string &filename);

struct Molecules {
    MoleculeSet ms;

    Molecules(const std::string &filename, bool read_hetatm, bool ignore_water, bool permissive_types);

    std::string input_file;

    [[nodiscard]] size_t length() const;
    [[nodiscard]] MoleculeSetStats info();
};

Molecules::Molecules(const std::string &filename, bool read_hetatm = true, bool ignore_water = true, bool permissive_types = false) {
    config::read_hetatm = read_hetatm;
    config::ignore_water = ignore_water;
    config::permissive_types = permissive_types;
    input_file = filename;
    
    ms = load_molecule_set(filename);
    if (ms.molecules().empty()) {
        throw std::runtime_error("No molecules were loaded from the input file");
    }
}

size_t Molecules::length() const {
    return ms.molecules().size();
}


std::vector<py::dict> get_available_methods() {
    std::vector<py::dict> results;
    std::string filename = fs::path(INSTALL_DIR) / "share" / "methods.json";
    using json = nlohmann::json;
    json j;
    std::ifstream f(filename);
    if (!f) {
        throw FileException(fmt::format("Cannot open file: {}", filename));
    }

    f >> j;
    f.close();

    for (const auto &method_info: j["methods"]) {
        std::optional<std::string> publication;
        if (!method_info["publication"].is_null()) {
            publication = method_info["publication"].get<std::string>();
        }

        results.emplace_back(py::dict(
            py::arg("name") = method_info["name"].get<std::string>(),
            py::arg("internal_name") = method_info["internal_name"].get<std::string>(),
            py::arg("full_name") = method_info["full_name"].get<std::string>(),
            py::arg("publication") = publication,
            py::arg("type") = method_info["type"].get<std::string>(),
            py::arg("has_parameters") = method_info["has_parameters"].get<bool>()
        ));
    }

    return results;
}


MoleculeSetStats Molecules::info() {
    ms.classify_atoms(AtomClassifier::PLAIN);
    return ms.get_stats();
}

py::dict atom_type_count_to_dict(const MoleculeSetStats::AtomTypeCount &atom_type_count) {
        return py::dict(
            py::arg("symbol") = atom_type_count.symbol,
            py::arg("count") = atom_type_count.count
    );
}

py::dict molecule_info_to_dict(const MoleculeSetStats &stats) {
    py::list atom_types_list;
    for (auto &count : stats.atom_type_counts) {
        atom_types_list.append(atom_type_count_to_dict(count));
    }

    return py::dict(
        py::arg("total_molecules") = stats.total_molecules,
        py::arg("total_atoms") = stats.total_atoms,
        py::arg("atom_type_counts") = atom_types_list
    );
}

std::vector<std::string> get_available_parameters(const std::string &method_name) {
    std::vector<std::string> parameters;
    for (const auto &parameter_file: get_parameter_files()) {
        if (not to_lowercase(parameter_file.filename().string()).starts_with(method_name)) {
            continue;
        }

        auto p = std::make_unique<Parameters>(parameter_file);
        if (method_name == p->method_name()) {
            parameters.emplace_back(parameter_file.stem().string());
        }
    }
    return parameters;
}

std::vector<std::tuple<std::string, std::vector<std::string>>> get_sutaible_methods_python(struct Molecules &molecules) {
    return get_suitable_methods(molecules.ms, molecules.ms.has_proteins(), false);    
}

py::dict get_parameters_metadata(const std::string &parameters_name) {
    auto normalized_params_name = parameters_name;
    if (normalized_params_name.ends_with(".json")) {
        // allow to provide name with or without extension
        normalized_params_name = normalized_params_name.substr(0, normalized_params_name.size() - 5);
    }

    std::string parameters_file = fs::path(INSTALL_DIR) / "share" / "parameters" / (normalized_params_name + ".json");
    using json = nlohmann::json;
    json j;
    std::ifstream f(parameters_file);
    if (!f) {
        throw FileException(fmt::format("Cannot open file: {}", parameters_file));
    }

    f >> j;
    f.close();

    return py::dict(
        py::arg("name") = j["metadata"]["name"].get<std::string>(),
        py::arg("publication") = j["metadata"]["publication"].get<std::string>(),
        py::arg("internal_name") = normalized_params_name
    );
}


std::map<std::string, std::vector<double>>
calculate_charges(struct Molecules &molecules, const std::string &method_name, std::optional<const std::string> &parameters_name, std::optional<const std::string> &chg_out_dir) {
    config::chg_out_dir = chg_out_dir.value_or(".");

    std::string method_file = fs::path(INSTALL_DIR) / "lib" / ("lib" + method_name + ".so");
    auto handle = dlopen(method_file.c_str(), RTLD_LAZY);

    auto get_method_handle = reinterpret_cast<Method *(*)()>(dlsym(handle, "get_method"));
    if (!get_method_handle) {
        throw std::runtime_error(dlerror());
    }

    auto method = (*get_method_handle)();

    molecules.ms.fulfill_requirements(method->get_requirements());

    std::unique_ptr<Parameters> parameters;
    if (method->has_parameters()) {
        if (!parameters_name.has_value()) {
            throw std::runtime_error(std::string("Method ") + method_name + std::string(" requires parameters"));
        }

        std::string parameter_file = fs::path(INSTALL_DIR) / "share" / "parameters" / (parameters_name.value() + ".json");
        if (not parameter_file.empty()) {
            parameters = std::make_unique<Parameters>(parameter_file);
            auto unclassified = molecules.ms.classify_set_from_parameters(*parameters, false, true);
            if (unclassified) {
                throw std::runtime_error("Selected parameters doesn't cover the whole molecule set");
            }
        }
    }

    method->set_parameters(parameters.get());

    /* Use only default values */
    for (const auto &[opt, info]: method->get_options()) {
        method->set_option_value(opt, info.default_value);
    }

    auto idk = Charges(); // TODO: rename to something coherent
    std::map<std::string, std::vector<double>> charges;
    for (auto &mol: molecules.ms.molecules()) {
        auto results = method->calculate_charges(mol);
        if (std::any_of(results.begin(), results.end(), [](double chg) { return not isfinite(chg); })) {
            fmt::print("Incorrect values encoutened for: {}. Skipping molecule.\n", mol.name());
        } else {
            idk.insert(mol.name(), results);
            charges[mol.name()] = results;
        }
    }
    
    save_charges(molecules, idk, molecules.input_file);

    dlclose(handle);

    return charges;
}

void save_charges(const Molecules &molecules, const Charges &charges, const std::string &filename) {
    std::filesystem::path dir(config::chg_out_dir);
    auto file_path = std::filesystem::path(filename);
    auto ext = file_path.extension().string();
    
    config::input_file = filename;
    CIF().save_charges(molecules.ms, charges, molecules.input_file);
    
    auto txt_str = file_path.filename().string() + ".txt"; 
    TXT().save_charges(molecules.ms, charges, dir / std::filesystem::path(txt_str));

    if (molecules.ms.has_proteins()) {
        auto pqr_str = file_path.filename().string() + ".pqr";
        PQR().save_charges(molecules.ms, charges, dir / std::filesystem::path(pqr_str));
    } else {
        auto mol2_str = file_path.filename().string() + ".mol2";
        Mol2().save_charges(molecules.ms, charges, dir / std::filesystem::path(mol2_str));
    }
}


PYBIND11_MODULE(chargefw2, m) {
    m.doc() = "Python bindings to ChargeFW2";
    py::class_<MoleculeSetStats::AtomTypeCount>(m, "AtomTypeCount")
        .def(py::init<>())
        .def_readwrite("symbol", &MoleculeSetStats::AtomTypeCount::symbol)
        .def_readwrite("count", &MoleculeSetStats::AtomTypeCount::count)
        .def("to_dict", [](const MoleculeSetStats::AtomTypeCount &self) {
            return atom_type_count_to_dict(self);
        });

    py::class_<MoleculeSetStats>(m, "MoleculeSetStats")
        .def(py::init<>())
        .def_readwrite("total_molecules", &MoleculeSetStats::total_molecules)
        .def_readwrite("total_atoms", &MoleculeSetStats::total_atoms)
        .def_readwrite("atom_type_counts", &MoleculeSetStats::atom_type_counts)
        .def("to_dict", [](const MoleculeSetStats &self) {
            return molecule_info_to_dict(self);
        });

    py::class_<Molecules>(m, "Molecules")
        .def(py::init<const std::string &, bool, bool, bool>(), py::arg("input_file"), py::arg("read_hetatm") = true,
                py::arg("ignore_water") = false, py::arg("permissive_types") = false)
        .def("__len__", &Molecules::length)
        .def("info", &Molecules::info);

    m.def("get_available_methods", &get_available_methods, "Return the list of all available methods");
    m.def("get_available_parameters", &get_available_parameters, "method_name"_a,
          "Return the list of all parameters of a given method");
    m.def("get_suitable_methods", &get_sutaible_methods_python, "molecules"_a, "Get methods and parameters that are suitable for a given set of molecules");
    m.def("get_parameters_metadata", &get_parameters_metadata, "parameters_name"_a);
    m.def("calculate_charges", &calculate_charges, "molecules"_a, "method_name"_a, py::arg("parameters_name") = py::none(), py::arg("chg_out_dir") = py::none(),
          "Calculate partial atomic charges for a given molecules and method", py::call_guard<py::gil_scoped_release>());
}
