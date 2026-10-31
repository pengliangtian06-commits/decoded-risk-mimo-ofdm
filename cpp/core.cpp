#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace py = pybind11;
using Matrix = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;
using RealVector = Eigen::VectorXd;
using Clock = std::chrono::steady_clock;

// LP64 complex-double LAPACK ABI provided by Accelerate/OpenBLAS. std::complex
// storage is interleaved real/imaginary doubles on the tested toolchain.
extern "C" {
    void zpotrf_(const char*, const int*, std::complex<double>*, const int*, int*);
    void zpotrs_(const char*, const int*, const int*, const std::complex<double>*,
                 const int*, std::complex<double>*, const int*, int*);
    void zpotri_(const char*, const int*, std::complex<double>*, const int*, int*);
}

static double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Established Gauss-Radau monitor, O(k) continued-fraction evaluation.
// Invalid recurrence => no early termination. No claim of an unconditional
// floating-point certificate is made.
static double radau_upper(const std::vector<double>& diag,
                          const std::vector<double>& off,
                          double next_off, double lower,
                          double initial_rr, double gauss) {
    const double infinity = std::numeric_limits<double>::infinity();
    long double pivot = static_cast<long double>(diag[0]) - lower;
    for (std::size_t i = 1; i < diag.size(); ++i) {
        if (!(pivot > 0)) return infinity;
        pivot = diag[i] - lower - static_cast<long double>(off[i-1]) * off[i-1] / pivot;
    }
    if (!(pivot > 0)) return infinity;
    long double denom = lower + static_cast<long double>(next_off) * next_off / pivot;
    for (int i = static_cast<int>(diag.size()) - 1; i >= 0; --i) {
        if (!(denom > 0)) return infinity;
        double coupling = i == static_cast<int>(diag.size()) - 1 ? next_off : off[i];
        denom = diag[i] - static_cast<long double>(coupling) * coupling / denom;
    }
    if (!(denom > 0)) return infinity;
    long double upper = initial_rr / denom;
    double floor = 64 * std::numeric_limits<double>::epsilon()
        * std::max({static_cast<double>(std::abs(upper)), std::abs(gauss), 1.0});
    double gap = static_cast<double>(upper - gauss);
    return (gap < -floor || !std::isfinite(gap)) ? infinity : std::max(gap, floor);
}

struct SolveResult {
    Matrix symbols;
    Eigen::MatrixXd approximate_gain;
    std::vector<int> iterations;
    std::vector<std::string> reasons;
    std::vector<double> residuals, energy_bounds;
    double seconds;
};

class PreparedSystem {
    Matrix h_, a_, transformed_, lapack_factor_;
    RealVector scale_, diagonal_;
    Eigen::LLT<Matrix> factor_;
    std::string backend_;
    double variance_, lower_, trace_, statistical_lower_, preparation_seconds_;
public:
    PreparedSystem(const Matrix& channel, double effective_variance, const std::string& backend)
        : backend_(backend), variance_(effective_variance) {
        auto start = Clock::now();
        if (backend != "pcg" && backend != "cg" && backend != "cholesky" &&
                backend != "lapack_cholesky" && backend != "matrixfree_pcg")
            throw std::invalid_argument("Unknown detector backend");
        if (channel.rows() < channel.cols() || channel.cols() < 1 || !channel.allFinite()
                || !(variance_ > 0) || !std::isfinite(variance_))
            throw std::invalid_argument("Invalid channel dimensions or effective variance");
        h_ = channel;
        if (backend == "matrixfree_pcg") {
            diagonal_ = h_.colwise().squaredNorm().transpose();
            diagonal_.array() += variance_;
        } else {
            a_.noalias() = h_.adjoint() * h_;
            a_.diagonal().array() += variance_;
            diagonal_ = a_.diagonal().real();
        }
        trace_ = diagonal_.sum();
        statistical_lower_ = variance_ * diagonal_.cwiseInverse().sum();
        if (backend == "lapack_cholesky") {
            lapack_factor_ = a_;
            const int n = a_.rows(); int info;
            const char lower = 'L';
            zpotrf_(&lower, &n, lapack_factor_.data(), &n, &info);
            if (info != 0) throw std::runtime_error("LAPACK Cholesky factorization failed");
        } else if (backend == "cholesky") {
            factor_.compute(a_);
            if (factor_.info() != Eigen::Success) throw std::runtime_error("Cholesky factorization failed");
        } else {
            if (backend != "cg") scale_ = diagonal_.array().sqrt().matrix();
            else scale_ = RealVector::Ones(h_.cols());
            if (backend != "matrixfree_pcg") {
                transformed_ = a_;
                for (Eigen::Index i = 0; i < a_.rows(); ++i)
                    for (Eigen::Index j = 0; j < a_.cols(); ++j)
                        transformed_(i,j) /= scale_(i) * scale_(j);
            }
            lower_ = 0.99 * variance_ / scale_.array().square().maxCoeff();
        }
        preparation_seconds_ = seconds_since(start);
    }

    SolveResult solve_native(const Matrix& received, const std::string& mode, double threshold,
                   int max_iterations, int min_iterations, int qam_order,
                   bool track_diagonal_polynomial=false) const {
        if (track_diagonal_polynomial && backend_ != "cg")
            throw std::invalid_argument("Published diagonal-polynomial baseline requires ordinary CG");
        if (received.rows() != h_.rows() || !received.allFinite() || threshold < 0
                || !std::isfinite(threshold) || max_iterations < 1 || min_iterations < 1)
            throw std::invalid_argument("Invalid observations or stopping configuration");
        if (mode != "fixed" && mode != "relative_residual" && mode != "noise_residual"
                && mode != "radau" && mode != "euclidean_bound" && mode != "decision_residual")
            throw std::invalid_argument("Unknown stopping mode");
        const int side = static_cast<int>(std::sqrt(qam_order));
        if (side*side != qam_order || side < 2 || (side & (side-1)))
            throw std::invalid_argument("QAM order must be a square power of two");
        const double normalization = std::sqrt(2.0*(qam_order-1)/3.0);
        const int n = h_.cols(), vectors = received.cols();
        Matrix solution(n, vectors);
        Eigen::MatrixXd approximate_gain;
        if (track_diagonal_polynomial) approximate_gain = Eigen::MatrixXd::Zero(n, vectors);
        std::vector<int> iterations(vectors);
        std::vector<std::string> reasons(vectors);
        std::vector<double> residuals(vectors), energy_bounds(vectors);
        double elapsed;
        {
            auto start = Clock::now();
            Matrix rhs = h_.adjoint() * received;
            if (backend_ == "cholesky" || backend_ == "lapack_cholesky") {
                if (backend_ == "lapack_cholesky") {
                    solution = rhs;
                    const char lower = 'L'; int info;
                    zpotrs_(&lower, &n, &vectors, lapack_factor_.data(), &n,
                            solution.data(), &n, &info);
                    if (info != 0) throw std::runtime_error("LAPACK Cholesky solve failed");
                } else solution = factor_.solve(rhs);
                for (int j = 0; j < vectors; ++j) {
                    iterations[j] = 0;
                    reasons[j] = "direct";
                    // No recursive residual exists for a direct solve. Do not
                    // charge it an extra diagnostic matrix multiply in timing.
                    residuals[j] = std::numeric_limits<double>::quiet_NaN();
                    energy_bounds[j] = 0;
                }
            } else for (int j = 0; j < vectors; ++j) {
                const Vector b = rhs.col(j);
                const double bnorm = b.norm();
                if (bnorm == 0) {
                    solution.col(j).setZero();
                    iterations[j] = 0; reasons[j] = "zero_rhs";
                    residuals[j] = 0; energy_bounds[j] = 0;
                    continue;
                }
                Vector z = Vector::Zero(n), r = b.cwiseQuotient(scale_), p = r;
                RealVector polynomial, polynomial_direction;
                if (track_diagonal_polynomial) {
                    polynomial = RealVector::Zero(n);
                    polynomial_direction = RealVector::Ones(n);
                }
                double rr = r.squaredNorm(), initial_rr = rr;
                double prev_alpha = 0, prev_beta = 0, gauss = 0;
                double upper = std::numeric_limits<double>::infinity(), original_rr = 0;
                std::vector<double> diagonal, off;
                std::vector<int> previous_decisions(2*n, -1000);
                int stable_decisions = 0;
                if (mode == "radau") { diagonal.reserve(max_iterations); off.reserve(max_iterations); }
                int completed = 0;
                reasons[j] = "max_iterations";
                for (int k = 0; k < max_iterations; ++k) {
                    Vector ap;
                    if (backend_ == "matrixfree_pcg") {
                        Vector unscaled = p.cwiseQuotient(scale_);
                        Vector observation = h_ * unscaled;
                        ap = (h_.adjoint() * observation + variance_*unscaled).cwiseQuotient(scale_);
                    } else ap.noalias() = transformed_ * p;
                    double curvature = std::real(p.dot(ap));
                    if (!(curvature > 0) || !std::isfinite(curvature)) {
                        reasons[j] = "numerical_breakdown";
                        break;
                    }
                    double alpha = rr / curvature;
                    z.noalias() += alpha * p;
                    r.noalias() -= alpha * ap;
                    double next_rr = r.squaredNorm(), beta = next_rr / rr;
                    if (track_diagonal_polynomial) {
                        // Companion form of Yin et al. (2014), Eq. (13).
                        // Replace A by diag(A) only in the equalizer polynomial,
                        // while alpha/beta retain the actual CG trajectory.
                        polynomial += alpha * polynomial_direction;
                        polynomial_direction = RealVector::Ones(n)
                            - diagonal_.cwiseProduct(polynomial) + beta * polynomial_direction;
                    }
                    completed = k + 1;
                    original_rr = r.cwiseProduct(scale_).squaredNorm();
                    if (mode == "radau") {
                        diagonal.push_back(k == 0 ? 1/alpha : 1/alpha + prev_beta/prev_alpha);
                        if (k > 0) off.push_back(std::sqrt(prev_beta) / prev_alpha);
                        gauss += alpha * rr;
                        upper = radau_upper(diagonal, off, std::sqrt(beta)/alpha,
                                            lower_, initial_rr, gauss);
                    }
                    if (std::sqrt(original_rr) / bnorm <= 1e-13) {
                        reasons[j] = "numerical_convergence";
                        break;
                    }
                    bool stop = false;
                    if (completed >= min_iterations) {
                        if (mode == "fixed") stop = completed >= static_cast<int>(threshold);
                        else if (mode == "relative_residual") stop = original_rr <= threshold*threshold*bnorm*bnorm;
                        else if (mode == "noise_residual") stop = original_rr <= threshold*variance_*trace_;
                        else if (mode == "radau") stop = upper <= threshold*variance_*n;
                        else if (mode == "euclidean_bound") stop = original_rr/(variance_*variance_) <= threshold*statistical_lower_;
                    }
                    if (mode == "decision_residual") {
                        double score = 0;
                        bool unchanged = true;
                        for (int u = 0; u < n; ++u) {
                            double gain = std::max(1-variance_/diagonal_(u), 1e-3);
                            auto value = z(u) / (scale_(u)*gain);
                            double real_value = value.real(), imag_value = value.imag();
                            int real_index = std::clamp(static_cast<int>(std::floor((real_value*normalization+side)/2)), 0, side-1);
                            int imag_index = std::clamp(static_cast<int>(std::floor((imag_value*normalization+side)/2)), 0, side-1);
                            unchanged = unchanged && real_index == previous_decisions[2*u] && imag_index == previous_decisions[2*u+1];
                            previous_decisions[2*u] = real_index; previous_decisions[2*u+1] = imag_index;
                            int bound_max = side/2-1;
                            double real_boundary = 2*std::clamp(std::round(real_value*normalization/2), -static_cast<double>(bound_max), static_cast<double>(bound_max))/normalization;
                            double imag_boundary = 2*std::clamp(std::round(imag_value*normalization/2), -static_cast<double>(bound_max), static_cast<double>(bound_max))/normalization;
                            double margin = std::min(std::abs(real_value-real_boundary), std::abs(imag_value-imag_boundary));
                            double proxy = variance_ / (diagonal_(u)*gain*gain);
                            double sensitivity = proxy / (proxy + margin*margin);
                            score += std::norm(r(u))*scale_(u)*scale_(u)/diagonal_(u)*sensitivity;
                        }
                        stable_decisions = unchanged ? stable_decisions+1 : 0;
                        stop = completed >= min_iterations && stable_decisions >= 1
                            && original_rr <= 0.04*bnorm*bnorm && score <= threshold*variance_*n;
                    }
                    if (stop) { reasons[j] = "stopping_rule"; break; }
                    p = r + beta * p;
                    rr = next_rr;
                    prev_alpha = alpha; prev_beta = beta;
                }
                solution.col(j) = z.cwiseQuotient(scale_);
                iterations[j] = completed;
                residuals[j] = original_rr;
                energy_bounds[j] = upper;
                if (track_diagonal_polynomial)
                    approximate_gain.col(j) = polynomial.array() * (diagonal_.array()-variance_);
            }
            elapsed = seconds_since(start);
        }
        return {std::move(solution), std::move(approximate_gain), std::move(iterations), std::move(reasons),
                std::move(residuals), std::move(energy_bounds), elapsed};
    }
    py::dict solve(const Matrix& received, const std::string& mode, double threshold,
                   int max_iterations, int min_iterations, int qam_order) const {
        SolveResult native;
        {
            py::gil_scoped_release release;
            native = solve_native(received, mode, threshold, max_iterations, min_iterations, qam_order);
        }
        py::dict result;
        result["symbols"] = native.symbols;
        result["iterations"] = native.iterations;
        result["stop_reasons"] = native.reasons;
        result["recursive_residual_norm2"] = native.residuals;
        result["radau_energy_upper"] = native.energy_bounds;
        result["solve_seconds"] = native.seconds;
        result["preparation_seconds"] = preparation_seconds_;
        return result;
    }
    Matrix system() const {
        if (a_.size()) return a_;
        Matrix a = h_.adjoint() * h_;
        a.diagonal().array() += variance_;
        return a;
    }
    double preparation_seconds() const { return preparation_seconds_; }
    RealVector inverse_diagonal() const {
        if (backend_ == "lapack_cholesky") {
            Matrix inverse = lapack_factor_;
            const int n = inverse.rows(); const char lower = 'L'; int info;
            zpotri_(&lower, &n, inverse.data(), &n, &info);
            if (info != 0) throw std::runtime_error("LAPACK Cholesky inverse failed");
            return inverse.diagonal().real();
        }
        if (backend_ != "cholesky") throw std::invalid_argument("Exact diagonal requires Cholesky backend");
        // A=L L^H => diag(A^-1)_j = ||L^-1(:,j)||^2. One triangular
        // solve avoids computing the second half of a full inverse solely
        // for its diagonal. Applies equally to direct and sparse anchors.
        Matrix inverse_lower = factor_.matrixL().solve(Matrix::Identity(h_.cols(), h_.cols()));
        return inverse_lower.colwise().squaredNorm().transpose();
    }
    RealVector diagonal() const { return diagonal_; }
};

// Batch all tones inside native code: Python per-tone dispatch is not charged
// selectively to a direct baseline. Input/output conversion remains included
// in the outer Python receiver wall-clock total for every backend.
static py::dict detect_tones(const std::vector<Matrix>& channels,
                            const std::vector<Matrix>& observations,
                            double variance, const std::string& backend,
                            const std::string& mode, double threshold,
                            int max_iterations, int qam_order,
                            const std::string& soft, int stride) {
    if (channels.empty() || channels.size() != observations.size() || stride < 1 ||
        (soft != "exact" && soft != "interpolate" && soft != "diagonal" && soft != "yin_approx"))
        throw std::invalid_argument("Invalid batch or soft-calibration grid");
    const int tones = channels.size();
    const int users = channels[0].cols();
    const int antennas = channels[0].rows(), vectors = observations[0].cols();
    std::vector<Matrix> solutions; solutions.reserve(tones);
    std::vector<Eigen::MatrixXd> approximate_gains;
    std::vector<int> iterations;
    std::vector<std::string> reasons;
    std::vector<double> residuals;
    Matrix inverse_diagonal = Matrix::Constant(users, tones,
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0));
    double preparation = 0, solving = 0, calibration = 0, total;
    {
        py::gil_scoped_release release;
        auto begin = Clock::now();
        for (int tone = 0; tone < tones; ++tone) {
            if (channels[tone].rows() != antennas || channels[tone].cols() != users ||
                observations[tone].rows() != antennas || observations[tone].cols() != vectors)
                throw std::invalid_argument("Inconsistent tone dimensions");
            PreparedSystem prepared(channels[tone], variance, backend);
            preparation += prepared.preparation_seconds();
            auto result = prepared.solve_native(observations[tone], mode, threshold,
                                                 max_iterations, 2, qam_order, soft=="yin_approx");
            solving += result.seconds;
            solutions.push_back(std::move(result.symbols));
            if (soft == "yin_approx") approximate_gains.push_back(std::move(result.approximate_gain));
            iterations.insert(iterations.end(), result.iterations.begin(), result.iterations.end());
            reasons.insert(reasons.end(), result.reasons.begin(), result.reasons.end());
            residuals.insert(residuals.end(), result.residuals.begin(), result.residuals.end());
            auto tick = Clock::now();
            if (soft == "yin_approx") {
                // Eq. (15) gives unbiased variance lambda/G_ii. No inverse
                // diagonal is computed for this published approximate baseline.
            } else if (soft == "diagonal") {
                inverse_diagonal.col(tone) = prepared.diagonal().cwiseInverse().cast<std::complex<double>>();
            } else if (soft == "exact" || tone % stride == 0) {
                if (backend == "cholesky" || backend == "lapack_cholesky") inverse_diagonal.col(tone) = prepared.inverse_diagonal().cast<std::complex<double>>();
                else {
                    PreparedSystem calibration_system(channels[tone], variance, "lapack_cholesky");
                    inverse_diagonal.col(tone) = calibration_system.inverse_diagonal().cast<std::complex<double>>();
                }
            }
            calibration += seconds_since(tick);
        }
        total = seconds_since(begin);
    }
    py::dict result;
    result["symbols_by_tone"] = solutions;
    result["approximate_gains_by_tone"] = approximate_gains;
    result["inverse_diagonal"] = inverse_diagonal.real().eval();
    result["iterations"] = iterations;
    result["stop_reasons"] = reasons;
    result["recursive_residual_norm2"] = residuals;
    result["preparation_seconds"] = preparation;
    result["solve_seconds"] = solving;
    result["soft_calibration_seconds"] = calibration;
    result["native_total_seconds"] = total;
    return result;
}

// Layered normalized min-sum on the published QC base graph. This is an
// independently written decoder; base-graph tables are supplied by the pinned
// py3gpp component. Bit zero corresponds to a positive input LLR.
static py::dict decode_ldpc(const Eigen::MatrixXd& input,
                           const Eigen::MatrixXi& graph, int lifting,
                           int max_iterations, double scaling, bool early_stop) {
    if (lifting < 2 || max_iterations < 1 || !(scaling > 0 && scaling <= 1)
            || graph.rows() < 1 || graph.cols() <= graph.rows()
            || input.rows() != (graph.cols()-2)*lifting || input.cols() < 1)
        throw std::invalid_argument("Invalid QC-LDPC dimensions or decoder configuration");
    const int checks = graph.rows(), columns = graph.cols();
    const int information = (columns-checks)*lifting, full = columns*lifting;
    struct Edge { int column, shift; };
    std::vector<std::vector<Edge>> rows(checks);
    std::vector<int> offsets(checks+1, 0);
    for (int row = 0; row < checks; ++row) {
        for (int col = 0; col < columns; ++col)
            if (graph(row,col) >= 0) rows[row].push_back({col, graph(row,col)%lifting});
        if (rows[row].size() < 2) throw std::invalid_argument("Check-node degree below two");
        offsets[row+1] = offsets[row]+rows[row].size();
    }
    Eigen::MatrixXi decoded(information, input.cols());
    std::vector<int> used(input.cols()), syndromes(input.cols());
    auto start = Clock::now();
    {
        py::gil_scoped_release release;
        for (int block = 0; block < input.cols(); ++block) {
            std::vector<double> beliefs(full, 0), messages(offsets.back()*lifting, 0);
            for (int i = 0; i < input.rows(); ++i) {
                if (std::isnan(input(i,block))) throw std::invalid_argument("NaN input LLR");
                beliefs[i+2*lifting] = std::clamp(input(i,block), -1e6, 1e6);
            }
            bool satisfied = false;
            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                for (int row = 0; row < checks; ++row) {
                    const auto& edges = rows[row];
                    const int degree = edges.size();
                    std::vector<double> extrinsic(degree*lifting);
                    for (int edge = 0; edge < degree; ++edge) {
                        const auto entry = edges[edge];
                        for (int k = 0; k < lifting; ++k) {
                            int node = entry.column*lifting + (k+entry.shift)%lifting;
                            extrinsic[edge*lifting+k] = beliefs[node] - messages[(offsets[row]+edge)*lifting+k];
                        }
                    }
                    for (int k = 0; k < lifting; ++k) {
                        double minimum = std::numeric_limits<double>::infinity();
                        double second = minimum;
                        int position = -1, sign_product = 1;
                        for (int edge = 0; edge < degree; ++edge) {
                            double value = extrinsic[edge*lifting+k], magnitude = std::abs(value);
                            if (value < 0) sign_product = -sign_product;
                            if (magnitude < minimum) { second=minimum; minimum=magnitude; position=edge; }
                            else if (magnitude < second) second=magnitude;
                        }
                        for (int edge = 0; edge < degree; ++edge) {
                            double value = extrinsic[edge*lifting+k];
                            int sign = value < 0 ? -1 : 1;
                            double message = scaling*sign_product*sign*(edge == position ? second : minimum);
                            messages[(offsets[row]+edge)*lifting+k] = message;
                            const auto entry = edges[edge];
                            beliefs[entry.column*lifting+(k+entry.shift)%lifting] = value+message;
                        }
                    }
                }
                used[block] = iteration+1;
                satisfied = true;
                for (int row = 0; row < checks && satisfied; ++row)
                    for (int k = 0; k < lifting && satisfied; ++k) {
                        int parity = 0;
                        for (const auto& entry : rows[row])
                            parity ^= beliefs[entry.column*lifting+(k+entry.shift)%lifting] < 0;
                        if (parity) satisfied = false;
                    }
                if (early_stop && satisfied) break;
            }
            for (int i = 0; i < information; ++i) decoded(i,block) = beliefs[i] < 0;
            syndromes[block] = satisfied ? 0 : 1;
        }
    }
    py::dict result;
    result["bits"] = decoded;
    result["iterations"] = used;
    result["syndrome_failed"] = syndromes;
    result["decode_seconds"] = seconds_since(start);
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Single-thread Eigen C++ detector cores. No RF real-time or energy claim.";
    py::class_<PreparedSystem>(m, "PreparedSystem")
        .def(py::init<const Matrix&, double, const std::string&>(),
             py::arg("channel_mean"), py::arg("effective_variance"), py::arg("backend")="pcg")
        .def("solve", &PreparedSystem::solve, py::arg("received"),
             py::arg("mode")="relative_residual", py::arg("threshold")=1e-3,
             py::arg("max_iterations")=64, py::arg("min_iterations")=2,
             py::arg("qam_order")=16)
        .def_property_readonly("system", &PreparedSystem::system)
        .def_property_readonly("preparation_seconds", &PreparedSystem::preparation_seconds);
    m.attr("arithmetic") = "complex128";
    m.attr("parallelism") = "one Eigen thread";
    m.attr("blas") = "external BLAS (Apple Accelerate on macOS)";
    m.def("detect_tones", &detect_tones, py::arg("channels"), py::arg("observations"),
          py::arg("effective_variance"), py::arg("backend"), py::arg("mode"),
          py::arg("threshold"), py::arg("max_iterations"), py::arg("qam_order"),
          py::arg("soft"), py::arg("anchor_stride"));
    m.def("decode_ldpc", &decode_ldpc, py::arg("recovered_llr"), py::arg("base_graph"),
          py::arg("lifting"), py::arg("max_iterations")=20,
          py::arg("scaling")=0.75, py::arg("early_stop")=true);
}
