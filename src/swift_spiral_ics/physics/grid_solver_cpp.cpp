#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <vector>
#include <iostream> // For debugging

// Define the gravitational constant (km/s)^2 kpc / Msun
// This should be consistent with physics/constants.py G.value
const double G_val = 4.300788457221135e-06; 

namespace py = pybind11;

// Function to compute potential of a single ring at (R_src, z_src) at a point (R_dest, z_dest)
double potential_of_ring(double M_ring, double R_src, double z_src, double R_dest, double z_dest, double eps) {
    // Elliptic integral formula for potential of a ring
    // Phi = -G * M / (pi * sqrt( (R_src+R_dest)^2 + (z_src-z_dest)^2 )) * K(k)
    // Where k^2 = (4 * R_src * R_dest) / ( (R_src+R_dest)^2 + (z_src-z_dest)^2 )
    // Softening eps is added to denominator to avoid singularity

    double term1 = (R_src + R_dest);
    double term2 = (z_src - z_dest);
    double A_sq = term1 * term1 + term2 * term2 + eps * eps;

    double k_sq_numerator = 4 * R_src * R_dest;
    double k_sq_denominator = A_sq; // This is the (R_src+R_dest)^2 + (z_src-z_dest)^2 + eps^2
    
    double k_sq = 0.0;
    if (k_sq_denominator > 0) { // Avoid division by zero
        k_sq = k_sq_numerator / k_sq_denominator;
    }
    
    // Ensure k^2 is within [0, 1] for ellipk
    k_sq = std::max(0.0, std::min(1.0 - 1e-12, k_sq)); // Clip to prevent issues with 1.0

    if (A_sq <= 0 || k_sq_denominator == 0 || k_sq == 0) {
        // Handle point mass potential for R_src=0 or R_dest=0 or very far points
        double dist_sq = (R_src - R_dest) * (R_src - R_dest) + term2 * term2 + eps * eps;
        if (dist_sq > 0) {
            return -G_val * M_ring / std::sqrt(dist_sq);
        } else {
            return 0.0; // Avoid singularity if dist_sq is 0
        }
    }

    double K_val = std::comp_ellint_1(std::sqrt(k_sq)); // K(k) = comp_ellint_1(sqrt(k^2))

    return -G_val * M_ring / (M_PI * std::sqrt(A_sq)) * K_val;
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

        nR_ = buf_R.shape[0];
        nz_ = buf_z.shape[0];

        R_grid_ = std::vector<double>(buf_R.ptr, buf_R.ptr + nR_);
        z_grid_ = std::vector<double>(buf_z.ptr, buf_z.ptr + nz_);

        rho_grid_ = py::array_t<double>({nR_, nz_});
        Phi_grid_ = py::array_t<double>({nR_, nz_});
        FR_grid_ = py::array_t<double>({nR_, nz_});
        FZ_grid_ = py::array_t<double>({nR_, nz_});
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

        for (auto item : all_pos_dict) {
            std::string comp_name = item.first.cast<std::string>();
            py::array_t<double> pos_py = item.second.cast<py::array_t<double>>();
            py::array_t<double> mass_py = all_mass_dict[item.first].cast<py::array_t<double>>();

            py::buffer_info buf_pos = pos_py.request();
            py::buffer_info buf_mass = mass_py.request();

            if (buf_pos.ndim != 2 || buf_pos.shape[1] != 3)
                throw std::runtime_error("Position arrays must be (N,3) dimensional!");
            if (buf_mass.ndim != 1)
                throw std::runtime_error("Mass arrays must be 1-dimensional!");
            if (buf_pos.shape[0] != buf_mass.shape[0])
                throw std::runtime_error("Position and mass arrays must have same N!");

            size_t n_particles = buf_pos.shape[0];
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

        // Manual binning for now. Can be optimized.
        // We will sum mass into cells and then divide by volume
        std::vector<std::vector<double>> mass_in_bins(nR_, std::vector<double>(nz_, 0.0));
        std::vector<std::vector<double>> volume_elements(nR_, std::vector<double>(nz_, 0.0)); // To store actual volume of bins

        for (size_t p_idx = 0; p_idx < r_all.size(); ++p_idx) {
            double R_val = r_all[p_idx];
            double z_val = z_all[p_idx];
            double m_val = mass_all[p_idx];

            // Find bin for R
            size_t iR = 0;
            while (iR < nR_ - 1 && R_val >= R_grid_[iR + 1]) {
                iR++;
            }
            if (iR == nR_ -1 && R_val > R_grid_[nR_ -1]) iR = nR_-1; // Handle edge case for R_grid max

            // Find bin for z
            size_t iz = 0;
            while (iz < nz_ - 1 && z_val >= z_grid_[iz + 1]) {
                iz++;
            }
             if (iz == nz_ -1 && z_val > z_grid_[nz_ -1]) iz = nz_-1; // Handle edge case for z_grid max
            
            // Check if within bounds
            if (R_val >= R_grid_[0] && R_val < R_grid_[nR_ - 1] &&
                z_val >= z_grid_[0] && z_val < z_grid_[nz_ - 1]) {
                 mass_in_bins[iR][iz] += m_val;
            }
        }
        
        // Convert mass_in_bins to rho_grid
        for (size_t i = 0; i < nR_; ++i) {
            double R_center = R_grid_[i];
            double dR_bin = (i < nR_ - 1) ? (R_grid_[i+1] - R_grid_[i]) : ((i > 0) ? (R_grid_[i] - R_grid_[i-1]) : 0.0);
            if (dR_bin == 0.0 && nR_ > 1) dR_bin = (R_grid_[nR_-1] - R_grid_[0]) / (nR_-1); // Fallback for single bin
            
            for (size_t j = 0; j < nz_; ++j) {
                double dZ_bin = (j < nz_ - 1) ? (z_grid_[j+1] - z_grid_[j]) : ((j > 0) ? (z_grid_[j] - z_grid_[j-1]) : 0.0);
                if (dZ_bin == 0.0 && nz_ > 1) dZ_bin = (z_grid_[nz_-1] - z_grid_[0]) / (nz_-1); // Fallback for single bin

                if (R_center > 0 && dR_bin > 0 && dZ_bin > 0) {
                    double volume = 2 * M_PI * R_center * dR_bin * dZ_bin;
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

        std::fill(Phi_grid_rw, Phi_grid_rw + nR_ * nz_, 0.0); // Reset Phi_grid

        // Loop over source grid points (mass rings)
        for (size_t i_src = 0; i_src < nR_; ++i_src) {
            double R_src = R_grid_[i_src];
            double dR_src = (i_src < nR_ - 1) ? (R_grid_[i_src+1] - R_grid_[i_src]) : ((i_src > 0) ? (R_grid_[i_src] - R_grid_[i_src-1]) : 0.0);
            if (dR_src == 0.0 && nR_ > 1) dR_src = (R_grid_[nR_-1] - R_grid_[0]) / (nR_-1);
            
            for (size_t j_src = 0; j_src < nz_; ++j_src) {
                double z_src = z_grid_[j_src];
                double dZ_src = (j_src < nz_ - 1) ? (z_grid_[j_src+1] - z_grid_[j_src]) : ((j_src > 0) ? (z_grid_[j_src] - z_grid_[j_src-1]) : 0.0);
                if (dZ_src == 0.0 && nz_ > 1) dZ_src = (z_grid_[nz_-1] - z_grid_[0]) / (nz_-1);
                
                double rho_src = rho_grid_r[i_src * nz_ + j_src];
                
                // Mass of this cell (approximation for ring mass)
                double mass_element = rho_src * (2 * M_PI * R_src * dR_src * dZ_src);
                
                if (rho_src == 0.0 || mass_element == 0.0) {
                    continue;
                }

                // Potential of a ring at (R_src, z_src) evaluated at (R_dest, z_dest)
                for (size_t i_dest = 0; i_dest < nR_; ++i_dest) {
                    double R_dest = R_grid_[i_dest];
                    for (size_t j_dest = 0; j_dest < nz_; ++j_dest) {
                        double z_dest = z_grid_[j_dest];
                        Phi_grid_rw[i_dest * nz_ + j_dest] += potential_of_ring(mass_element, R_src, z_src, R_dest, z_dest, eps_);
                    }
                }
            }
        }

        // Compute forces from potential using numerical differentiation (central difference)
        auto FR_grid_rw = FR_grid_.mutable_data();
        auto FZ_grid_rw = FZ_grid_.mutable_data();
        
        for (size_t i = 0; i < nR_; ++i) {
            for (size_t j = 0; j < nz_; ++j) {
                // dPhi/dR
                if (i > 0 && i < nR_ - 1) {
                    FR_grid_rw[i * nz_ + j] = -(Phi_grid_rw[(i + 1) * nz_ + j] - Phi_grid_rw[(i - 1) * nz_ + j]) / (R_grid_[i+1] - R_grid_[i-1]);
                } else {
                    // Boundary handling (forward/backward difference or zero)
                    FR_grid_rw[i * nz_ + j] = 0.0; // Simplistic
                }

                // dPhi/dz
                if (j > 0 && j < nz_ - 1) {
                    FZ_grid_rw[i * nz_ + j] = -(Phi_grid_rw[i * nz_ + (j + 1)] - Phi_grid_rw[i * nz_ + (j - 1)]) / (z_grid_[j+1] - z_grid_[j-1]);
                } else {
                    // Boundary handling
                    FZ_grid_rw[i * nz_ + j] = 0.0; // Simplistic
                }
            }
        }
    }

    // Get interpolated potential and forces
    py::dict get_potential_and_forces(py::array_t<double> R_py, py::array_t<double> z_py) {
        py::buffer_info buf_R = R_py.request();
        py::buffer_info buf_z = z_py.request();

        if (buf_R.ndim != 1 || buf_z.ndim != 1 || buf_R.shape[0] != buf_z.shape[0])
            throw std::runtime_error("R and z arrays must be 1D and of same size!");

        size_t N = buf_R.shape[0];
        const double* R_ptr = static_cast<double*>(buf_R.ptr);
        const double* z_ptr = static_cast<double*>(buf_z.ptr);

        py::array_t<double> Phi_interp({N});
        py::array_t<double> FR_interp({N});
        py::array_t<double> FZ_interp({N});

        auto Phi_interp_rw = Phi_interp.mutable_data();
        auto FR_interp_rw = FR_interp.mutable_data();
        auto FZ_interp_rw = FZ_interp.mutable_data();

        // Use linear interpolation for now
        // This is a placeholder, actual interpolation should be done carefully
        // For actual implementation, a 2D interpolation library should be used (e.g., from Python)
        // Or implement 2D linear interpolation here.
        
        // For now, let's just do nearest neighbor for quick test
        for (size_t p = 0; p < N; ++p) {
            double r_val = R_ptr[p];
            double z_val = z_ptr[p];

            // Find nearest R_grid point
            size_t iR = 0;
            double min_dist_R = std::abs(r_val - R_grid_[0]);
            for (size_t i = 1; i < nR_; ++i) {
                double dist = std::abs(r_val - R_grid_[i]);
                if (dist < min_dist_R) {
                    min_dist_R = dist;
                    iR = i;
                }
            }

            // Find nearest z_grid point
            size_t iz = 0;
            double min_dist_z = std::abs(z_val - z_grid_[0]);
            for (size_t j = 1; j < nz_; ++j) {
                double dist = std::abs(z_val - z_grid_[j]);
                if (dist < min_dist_z) {
                    min_dist_z = dist;
                    iz = j;
                }
            }
            
            Phi_interp_rw[p] = *(Phi_grid_.data() + iR * nz_ + iz);
            FR_interp_rw[p] = *(FR_grid_.data() + iR * nz_ + iz);
            FZ_interp_rw[p] = *(FZ_grid_.data() + iR * nz_ + iz);
        }
        
        py::dict result;
        result["Phi"] = Phi_interp;
        result["FR"] = FR_interp;
        result["FZ"] = FZ_interp;
        return result;
    }

private:
    double eps_;
    int nR_, nz_;
    std::vector<double> R_grid_;
    std::vector<double> z_grid_;
    
    py::array_t<double> rho_grid_;
    py::array_t<double> Phi_grid_;
    py::array_t<double> FR_grid_;
    py::array_t<double> FZ_grid_;

    // Galaxy parameters
    double m200_, c200_, m_bulge_, a_bulge_, M_disc_star_, R_d_star_, z_d_star_, M_disc_gas_, R_d_gas_, z_d_gas_;
};

// pybind11 module definition
PYBIND11_MODULE(_grid_solver_cpp, m) {
    m.doc() = "pybind11 plugin for C++ grid solver for galaxy ICs"; // optional module docstring

    py::class_<GridSolverCpp>(m, "GridSolverCpp")
        .def(py::init<
            py::array_t<double>, py::array_t<double>, double,
            double, double, double, double, double, double, double, double, double, double, double
        >(),
            py::arg("R_grid"), py::arg("z_grid"), py::arg("eps"),
            py::arg("m200"), py::arg("c200"), py::arg("m_bulge"), py::arg("a_bulge"),
            py::arg("M_disc_star"), py::arg("R_d_star"), py::arg("z_d_star"),
            py::arg("M_disc_gas"), py::arg("R_d_gas"), py::arg("z_d_gas")
        )
        .def("bin_particles_to_grid", &GridSolverCpp::bin_particles_to_grid,
             py::arg("all_pos_dict"), py::arg("all_mass_dict"))
        .def("compute_potential_grid", &GridSolverCpp::compute_potential_grid)
        .def("get_potential_and_forces", &GridSolverCpp::get_potential_and_forces,
             py::arg("R"), py::arg("z"));
}
