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

#include "structures/molecule_set.h"
#include "formats/reader.h"
#include "config.h"
#include "charges.h"
#include "candidates.h"
#include "utility/strings.h"


namespace fs = std::filesystem;
namespace py = pybind11;
using namespace pybind11::literals;


std::map<std::string, std::vector<double>>
calculate_charges(struct Molecules &molecules, const std::string &method_name, std::optional<const std::string> &parameters_name);

std::vector<std::string> get_available_methods();

std::vector<std::string> get_available_parameters(const std::string &method_name);

std::vector<std::tuple<std::string, std::vector<std::string>>> get_sutaible_methods_python(struct Molecules &molecules);

struct MoleculeInfo {
    size_t total_molecules;
    size_t total_atoms;

    struct AtomTypeCount {
        std::string symbol;
        std::string cls;
        std::string type;
        int count;

        [[nodiscard]] py::dict to_dict() const;
    };

    std::map<size_t, AtomTypeCount> atom_type_counts;

    [[nodiscard]] py::dict to_dict() const;
};

py::dict MoleculeInfo::to_dict() const {
    py::dict atom_types_dict;
    for (const auto& [key, value] : atom_type_counts) {
        atom_types_dict[py::cast(key)] = value.to_dict();
    }

    return py::dict(
        py::arg("total_molecules") = total_molecules,
        py::arg("total_atoms") = total_atoms,
        py::arg("atom_type_counts") = atom_types_dict
    );
}

py::dict MoleculeInfo::AtomTypeCount::to_dict() const {
        return py::dict(
            py::arg("symbol") = symbol,
            py::arg("cls") = cls,
            py::arg("type") = type,
            py::arg("count") = count
    );
}

struct Molecules {
    MoleculeSet ms;

    Molecules(const std::string &filename, bool read_hetatm, bool ignore_water);

    [[nodiscard]] size_t length() const;
    [[nodiscard]] MoleculeInfo info();
};

Molecules::Molecules(const std::string &filename, bool read_hetatm = true, bool ignore_water = true) {
    config::read_hetatm = read_hetatm;
    config::ignore_water = ignore_water;

    auto file = fopen("/tmp/atomic-logs/atomic-log.log", "a");
    fmt::print(file, "Reading molecules from file: {}\n", filename);
    auto start = std::chrono::high_resolution_clock::now();

    ms = load_molecule_set(filename);
    if (ms.molecules().empty()) {
        fclose(file);
        throw std::runtime_error("No molecules were loaded from the input file");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    fmt::print(file, "Reading molecules took: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    auto start_requirements = std::chrono::high_resolution_clock::now();
    fmt::print(file, "Fulfilling requirements\n");
    
    ms.fulfill_requirements({RequiredFeatures::DISTANCE_TREE, RequiredFeatures::BOND_DISTANCES});
    
    auto end_requirements = std::chrono::high_resolution_clock::now();
    fmt::print(file, "Fulfilling requirements took: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_requirements - start_requirements).count());
    
    fclose(file);
}


size_t Molecules::length() const {
    return ms.molecules().size();
}


std::vector<std::string> get_available_methods() {
    std::vector<std::string> results;
    std::string filename = fs::path(INSTALL_DIR) / "share" / "methods.json";
    using json = nlohmann::json;
    json j;
    std::ifstream f(filename);
    if (!f) {
        fmt::print(stderr, "Cannot open file: {}\n", filename);
        exit(EXIT_FILE_ERROR);
    }

    f >> j;
    f.close();

    for (const auto &method_info: j["methods"]) {
        auto method_name = method_info["internal_name"].get<std::string>();
        results.emplace_back(method_name);
    }

    return results;
}


MoleculeInfo Molecules::info() {
    ms.classify_atoms(AtomClassifier::PLAIN);

    MoleculeInfo result;
    std::map<size_t, int> counts;
    size_t n_atoms = 0;

    for (const auto &m: ms.molecules()) {
        for (auto &a : m.atoms()) {
            counts[a.type()] += 1;
            n_atoms++;
        }
    }

    result.total_molecules = length();
    result.total_atoms = n_atoms;

    auto atom_types = ms.atom_types();

    if (atom_types.size() > 0) {
        for (auto &[key, val] : counts) {
            auto [symbol, cls, type] = atom_types[key];
            
            MoleculeInfo::AtomTypeCount atom_type_count;
            atom_type_count.symbol = symbol;
            atom_type_count.cls = cls;
            atom_type_count.type = type;
            atom_type_count.count = val;
            
            result.atom_type_counts[key] = atom_type_count;
        }
    }
    
    return result;
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


std::map<std::string, std::vector<double>>
calculate_charges(struct Molecules &molecules, const std::string &method_name, std::optional<const std::string> &parameters_name) {
    auto total_start = std::chrono::high_resolution_clock::now();

    std::string method_file = fs::path(INSTALL_DIR) / "lib" / ("lib" + method_name + ".so");
    auto file = fopen("/tmp/atomic-logs/atomic-log.log", "a");
    auto dlopen_start = std::chrono::high_resolution_clock::now();
    auto handle = dlopen(method_file.c_str(), RTLD_LAZY);
    auto dlopen_end = std::chrono::high_resolution_clock::now();
    fmt::print(file, "dlopen time: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(dlopen_end - dlopen_start).count());


    auto get_method_handle = reinterpret_cast<Method *(*)()>(dlsym(handle, "get_method"));
    if (!get_method_handle) {
        throw std::runtime_error(dlerror());
    }

    auto method = (*get_method_handle)();
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

    fmt::print(file, "Setting params\n");
    method->set_parameters(parameters.get());
    fmt::print(file, "Set params\n");

    /* Use only default values */
    for (const auto &[opt, info]: method->get_options()) {
        fmt::print(file, "Setting option value\n");
        method->set_option_value(opt, info.default_value);
    }

    std::map<std::string, std::vector<double>> charges;
    auto calc_start = std::chrono::high_resolution_clock::now();
    for (auto &mol: molecules.ms.molecules()) {
        fmt::print(file, "Calculating charges for molecule: {}\n", mol.name());

        auto results = method->calculate_charges(mol);
        if (std::any_of(results.begin(), results.end(), [](double chg) { return not isfinite(chg); })) {
            fmt::print(file, "Incorrect values encoutened for: {}. Skipping molecule.\n", mol.name());
        } else {
            fmt::print(file, "calculated charges for molecule: {}\n", mol.name());
            charges[mol.name()] = results;
        }
    }
    auto calc_end = std::chrono::high_resolution_clock::now();
    fmt::print(file, "Total calculation time: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(calc_end - calc_start).count());

    dlclose(handle);

    auto total_end = std::chrono::high_resolution_clock::now();
    fmt::print(file, "Total time: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count());

    fclose(file);

    return charges;
}


PYBIND11_MODULE(chargefw2, m) {
    m.doc() = "Python bindings to ChargeFW2";
    py::class_<MoleculeInfo::AtomTypeCount>(m, "AtomTypeCount")
        .def(py::init<>())
        .def_readwrite("symbol", &MoleculeInfo::AtomTypeCount::symbol)
        .def_readwrite("cls", &MoleculeInfo::AtomTypeCount::cls)
        .def_readwrite("type", &MoleculeInfo::AtomTypeCount::type)
        .def_readwrite("count", &MoleculeInfo::AtomTypeCount::count)
        .def("to_dict", &MoleculeInfo::AtomTypeCount::to_dict);

    py::class_<MoleculeInfo>(m, "MoleculeInfo")
        .def(py::init<>())
        .def_readwrite("total_molecules", &MoleculeInfo::total_molecules)
        .def_readwrite("total_atoms", &MoleculeInfo::total_atoms)
        .def_readwrite("atom_type_counts", &MoleculeInfo::atom_type_counts)
        .def("to_dict", &MoleculeInfo::to_dict);

    py::class_<Molecules>(m, "Molecules")
        .def(py::init<const std::string &, bool, bool>(), py::arg("input_file"), py::arg("read_hetatm") = true,
                py::arg("ignore_water") = false)
        .def("__len__", &Molecules::length)
        .def("info", &Molecules::info);

    m.def("get_available_methods", &get_available_methods, "Return the list of all available methods");
    m.def("get_available_parameters", &get_available_parameters, "method_name"_a,
          "Return the list of all parameters of a given method");
    m.def("get_suitable_methods", &get_sutaible_methods_python, "molecules"_a, "Get methods and parameters that are suitable for a given set of molecules");
    m.def("calculate_charges", &calculate_charges, "molecules"_a, "method_name"_a, py::arg("parameters_name") = py::none(),
          "Calculate partial atomic charges for a given molecules and method", py::call_guard<py::gil_scoped_release>());
}
