#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For std::vector, std::string
#include <cmath>          // For std::sqrt, std::max, std::min
#include <vector>
#include <algorithm>      // For std::lower_bound
#include <stdexcept>      // For std::runtime_error

// Define constants
const double G_val = 4.300788457221135e-06; 
const double PI = 3.14159265358979323846;

namespace py = pybind11;

// Arithmetic-Geometric Mean for Elliptic Integral
double agm(double a, double b) {
    const double tol = 1e-9;
    while (std::abs(a - b) > tol) {
        double a_next = 0.5 * (a + b);
        double b_next = std::sqrt(a * b);
        a = a_next;
        b = b_next;
    }
    return a;
}

// Complete Elliptic Integral of the First Kind K(k)
double my_ellipk(double k) {
    // K(k) = pi / (2 * agm(1, sqrt(1 - k^2)))
    if (k >= 1.0) return 1e10; // Large number for singularity
    return PI / (2.0 * agm(1.0, std::sqrt(1.0 - k * k)));
}

// Function to compute potential of a single ring at (R_src, z_src) at a point (R_dest, z_dest)
double potential_of_ring(double M_ring, double R_src, double z_src, double R_dest, double z_dest, double eps) {
    double term1 = (R_src + R_dest);
    double term2 = (z_src - z_dest);
    double A_sq = term1 * term1 + term2 * term2 + eps * eps;

    double k_sq_numerator = 4 * R_src * R_dest;
    double k_sq = 0.0;
    if (A_sq > 0) {
        k_sq = k_sq_numerator / A_sq;
    }
    
    // Clip k_sq
    k_sq = std::max(0.0, std::min(1.0 - 1e-9, k_sq));

    if (k_sq == 0) {
        // Point mass limit (on axis)
        return -G_val * M_ring / std::sqrt(A_sq);
    }

    double K_val = my_ellipk(std::sqrt(k_sq));

    return -G_val * M_ring / (PI * std::sqrt(A_sq)) * K_val;
}


class GridSolverCpp {
public:
    // Constructor
    GridSolverCpp(
        py::array_t<double> R_grid_py, py::array_t<double> z_grid_py, double eps,
        double m200, double c200, double m_bulge, double a_bulge,
        double M_disc_star, double R_d_star, double z_d_star,
        double M_disc_gas, double R_d_gas, double z_d_gas
    ) : 
        eps_(eps), 
        m200_(m200), c200_(c200), m_bulge_(m_bulge), a_bulge_(a_bulge),
        M_disc_star_(M_disc_star), R_d_star_(R_d_star), z_d_star_(z_d_star),
        M_disc_gas_(M_disc_gas), R_d_gas_(R_d_gas), z_d_gas_(z_d_gas)
    {
        py::buffer_info buf_R = R_grid_py.request();
        py::buffer_info buf_z = z_grid_py.request();

        if (buf_R.ndim != 1 || buf_z.ndim != 1)
            throw std::runtime_error("Grid arrays must be 1-dimensional!");

        nR_ = static_cast<size_t>(buf_R.shape[0]);
        nz_ = static_cast<size_t>(buf_z.shape[0]);

        R_grid_ = std::vector<double>(static_cast<double*>(buf_R.ptr), static_cast<double*>(buf_R.ptr) + nR_);
        z_grid_ = std::vector<double>(static_cast<double*>(buf_z.ptr), static_cast<double*>(buf_z.ptr) + nz_);

        rho_grid_ = py::array_t<double>({static_cast<ssize_t>(nR_), static_cast<ssize_t>(nz_)});
        Phi_grid_ = py::array_t<double>({static_cast<ssize_t>(nR_), static_cast<ssize_t>(nz_)});
    }

    // Bin particles to grid
    void bin_particles_to_grid(
        const py::dict& all_pos_dict, 
        const py::dict& all_mass_dict
    ) {
        auto rho_grid_rw = rho_grid_.mutable_data();
        std::fill(rho_grid_rw, rho_grid_rw + nR_ * nz_, 0.0); // Reset rho_grid

        std::vector<double> r_all;
        std::vector<double> z_all;
        std::vector<double> mass_all;

        // Iterate through component dicts
        for (auto item : all_pos_dict) {
            py::array_t<double> pos_py = item.second.cast<py::array_t<double>>();
            py::array_t<double> mass_py = all_mass_dict[item.first].cast<py::array_t<double>>();

            py::buffer_info buf_pos = pos_py.request();
            py::buffer_info buf_mass = mass_py.request();

            size_t n_particles = static_cast<size_t>(buf_pos.shape[0]);
            const double* pos_ptr = static_cast<double*>(buf_pos.ptr);
            const double* mass_ptr = static_cast<double*>(buf_mass.ptr);

            for (size_t i = 0; i < n_particles; ++i) {
                double R_comp = std::sqrt(pos_ptr[i * 3 + 0] * pos_ptr[i * 3 + 0] + pos_ptr[i * 3 + 1] * pos_ptr[i * 3 + 1]);
                double z_comp = pos_ptr[i * 3 + 2];
                
                r_all.push_back(R_comp);
                z_all.push_back(z_comp);
                mass_all.push_back(mass_ptr[i]);
            }
        }

        if (r_all.empty()) return;

        // Manual binning: Sum mass into cells and then divide by volume
        std::vector<std::vector<double>> mass_in_bins(nR_, std::vector<double>(nz_, 0.0));

        for (size_t p_idx = 0; p_idx < r_all.size(); ++p_idx) {
            double R_val = r_all[p_idx];
            double z_val = z_all[p_idx];
            double m_val = mass_all[p_idx];

            // Find bin for R (nearest neighbor logic for binning)
            // Assuming R_grid is sorted. Using lower_bound to find index.
            // Note: R_grid are points. We bin based on proximity? Or assuming R_grid are bin edges?
            // The logic "R_center = R_grid[i]" suggests R_grid are centers.
            // Let's assume R_grid defines the grid points and we assign to the nearest.
            
            size_t iR = 0;
            if (R_val >= R_grid_[0]) {
                auto it = std::lower_bound(R_grid_.begin(), R_grid_.end(), R_val);
                iR = static_cast<size_t>(it - R_grid_.begin());
                if (iR == nR_) iR--; 
                // Check if previous point is closer
                if (iR > 0 && (R_val - R_grid_[iR-1] < R_grid_[iR] - R_val)) {
                    iR--;
                }
            }
            
            size_t iz = 0;
            if (z_val >= z_grid_[0]) {
                auto it = std::lower_bound(z_grid_.begin(), z_grid_.end(), z_val);
                iz = static_cast<size_t>(it - z_grid_.begin());
                if (iz == nz_) iz--;
                if (iz > 0 && (z_val - z_grid_[iz-1] < z_grid_[iz] - z_val)) {
                    iz--;
                }
            }
            
            // Check limits again
            if (iR < nR_ && iz < nz_) {
                mass_in_bins[iR][iz] += m_val;
            }
        }
        
        // Convert mass_in_bins to rho_grid
        for (size_t i = 0; i < nR_; ++i) {
            double R_center = R_grid_[i];
            
            // Estimate bin width
            double dR_bin_actual = 0.0;
            if (i < nR_ - 1) dR_bin_actual = R_grid_[i+1] - R_grid_[i];
            else if (i > 0) dR_bin_actual = R_grid_[i] - R_grid_[i-1];
            if (nR_ == 1) dR_bin_actual = 1.0; // dummy

            for (size_t j = 0; j < nz_; ++j) {
                double dZ_bin_actual = 0.0;
                if (j < nz_ - 1) dZ_bin_actual = z_grid_[j+1] - z_grid_[j];
                else if (j > 0) dZ_bin_actual = z_grid_[j] - z_grid_[j-1];
                if (nz_ == 1) dZ_bin_actual = 1.0;

                if (dR_bin_actual > 0 && dZ_bin_actual > 0) {
                    double volume;
                    if (R_center == 0.0) {
                        // Central cylinder volume: pi * (dR/2)^2 * h
                        double r_cyl = dR_bin_actual / 2.0;
                        volume = PI * r_cyl * r_cyl * dZ_bin_actual;
                    } else {
                        // Annulus volume: 2 * pi * R * dR * h
                        volume = 2 * PI * R_center * dR_bin_actual * dZ_bin_actual;
                    }
                    
                    if (volume > 0) {
                         rho_grid_rw[i * nz_ + j] = mass_in_bins[i][j] / volume;
                    }
                }
            }
        }
    }

    // Compute potential grid
    void compute_potential_grid() {
        auto Phi_grid_rw = Phi_grid_.mutable_data();
        auto rho_grid_r = rho_grid_.data();

        std::fill(Phi_grid_rw, Phi_grid_rw + nR_ * nz_, 0.0);

        for (size_t i_src = 0; i_src < nR_; ++i_src) {
            double R_src = R_grid_[i_src];
            
            double dR_src = 0.0;
            if (i_src < nR_ - 1) dR_src = R_grid_[i_src+1] - R_grid_[i_src];
            else if (i_src > 0) dR_src = R_grid_[i_src] - R_grid_[i_src-1];
            
            for (size_t j_src = 0; j_src < nz_; ++j_src) {
                double z_src = z_grid_[j_src];
                
                double dZ_src = 0.0;
                if (j_src < nz_ - 1) dZ_src = z_grid_[j_src+1] - z_grid_[j_src];
                else if (j_src > 0) dZ_src = z_grid_[j_src] - z_grid_[j_src-1];
                
                double rho_src = rho_grid_r[i_src * nz_ + j_src];
                
                double mass_element;
                if (R_src == 0.0) {
                    double r_cyl = dR_src / 2.0;
                    mass_element = rho_src * (PI * r_cyl * r_cyl * dZ_src);
                } else {
                    mass_element = rho_src * (2 * PI * R_src * dR_src * dZ_src);
                }
                
                if (rho_src == 0.0 || mass_element == 0.0) {
                    continue;
                }

                for (size_t i_dest = 0; i_dest < nR_; ++i_dest) {
                    double R_dest = R_grid_[i_dest];
                    for (size_t j_dest = 0; j_dest < nz_; ++j_dest) {
                        double z_dest = z_grid_[j_dest];
                        Phi_grid_rw[i_dest * nz_ + j_dest] += potential_of_ring(mass_element, R_src, z_src, R_dest, z_dest, eps_);
                    }
                }
            }
        }

        // Forces are computed in Python wrapper via splines
    }

    py::array_t<double> get_rho_grid() const { return rho_grid_; }
    py::array_t<double> get_Phi_grid() const { return Phi_grid_; }

private:
    double eps_;
    size_t nR_, nz_;
    std::vector<double> R_grid_;
    std::vector<double> z_grid_;
    
    py::array_t<double> rho_grid_;
    py::array_t<double> Phi_grid_;

    // Galaxy parameters
    double m200_, c200_, m_bulge_, a_bulge_, M_disc_star_, R_d_star_, z_d_star_, M_disc_gas_, R_d_gas_, z_d_gas_;
};

// pybind11 module definition
PYBIND11_MODULE(_grid_solver_cpp, m) {
    m.doc() = "pybind11 plugin for C++ grid solver for galaxy ICs";

    py::class_<GridSolverCpp>(m, "GridSolverCpp")
        .def(py::init<
            py::array_t<double>, py::array_t<double>, double,
            double, double, double, double, double, double, double, double, double, double
        >(),
            py::arg("R_grid"), py::arg("z_grid"), py::arg("eps"),
            py::arg("m200"), py::arg("c200"), py::arg("m_bulge"), py::arg("a_bulge"),
            py::arg("M_disc_star"), py::arg("R_d_star"), py::arg("z_d_star"),
            py::arg("M_disc_gas"), py::arg("R_d_gas"), py::arg("z_d_gas")
        )
        .def("bin_particles_to_grid", &GridSolverCpp::bin_particles_to_grid,
             py::arg("all_pos_dict"), py::arg("all_mass_dict"))
        .def("compute_potential_grid", &GridSolverCpp::compute_potential_grid)
        .def_property_readonly("rho_grid", &GridSolverCpp::get_rho_grid)
        .def_property_readonly("Phi_grid", &GridSolverCpp::get_Phi_grid);
}
