// #include "pdm_dict.h"
#include <Eigen/Dense>
#include <WignerSymbol.hpp>
#include <cassert>
#include <chrono>
#include <fmt/core.h>
#include <fstream>
#include <numeric>

namespace Format {
static const Eigen::IOFormat NumpyFmt(
    Eigen::StreamPrecision, // 流精度
    0, // 标志位（无特殊标志）
    " ", // 元素分隔符（空格）
    " \n", // 行分隔符（空格 + 换行）
    "[", // 行前缀
    "]", // 行后缀
    "[", // 矩阵前缀（每行前）
    "]" // 矩阵后缀（每行后）
);
}

// separate T^1 to T^1_(-1,0,1)
// only for T^1_(-1,0,1)!!!
inline const std::array<Eigen::MatrixXd, 3> spin_projection(const Eigen::MatrixXd& pdm, const int tjb, const int tjo, const int tjk, const int ibra, const int iket)
{
    assert(tjo == 2);
    auto ppdm
        = std::array<Eigen::MatrixXd, 3> { Eigen::MatrixXd::Zero(pdm.rows(), pdm.cols()),
              Eigen::MatrixXd::Zero(pdm.rows(), pdm.cols()),
              Eigen::MatrixXd::Zero(pdm.rows(), pdm.cols()) };
    /*
    <IS'Ms'|T^{1, M}|JSMs> =
    (-1)^{S' - Ms'} * |S'   1  S | <IS'||T^{1}||JS>
                      |-Ms' M  Ms|
    */
    // Note that 2*S, 2*Ms are used in the WignerSymbol library
    auto tmb = -tjb + 2 * ibra;
    auto tmk = -tjk + 2 * iket;
    for (int iop = 0; iop <= tjo; iop++) {
        auto tmo = -tjo + 2 * iop;

        // block2
        // auto factor = std::pow(-1, (tjb - tmb) / 2) * util::wigner_3j(tjb, tjo, tjk, -tmb, tmo, tmk);
        // CG
        auto factor = util::CG(tjk, tjo, tjb, tmk, tmo, tmb);
        // QCMaquis
        // auto factor = std::pow(-1, (tjk + tmb - tjo) / 2) * util::wigner_3j(tjk, tjo, tjb, tmk, tmo, tmb);
        if (factor != 0.0) {
            ppdm[iop] = factor * pdm;
        }
    }
    return ppdm;
}
// from T^1_(-1,0,1) to Tx, Ty, Tz
// Note that Tx, Ty, Tz are complex
inline const std::array<Eigen::MatrixXcd, 3> xyz_proj(const std::array<Eigen::MatrixXd, 3>& ppdm)
{
    /*
    (Tx, Ty, Tz) =
                              |-1/2 i/2  0|
    (T^1_(1), T^1_(-1), T^1_0)|1/2  i/2  0|
                              |0    0    1/sqrt(2)|
    */
    auto xpdm
        = std::array<Eigen::MatrixXcd, 3> { Eigen::MatrixXcd::Zero(ppdm[0].rows(), ppdm[0].cols()),
              Eigen::MatrixXcd::Zero(ppdm[0].rows(), ppdm[0].cols()),
              Eigen::MatrixXcd::Zero(ppdm[0].rows(), ppdm[0].cols()) };
    xpdm[0] = std::complex<double>(0.5, 0.0) * (ppdm[0] - ppdm[2]);
    xpdm[1] = std::complex<double>(0.0, 0.5) * (ppdm[0] + ppdm[2]);
    xpdm[2] = std::complex<double>(std::sqrt(0.5), 0) * ppdm[1];
    return xpdm;
}

std::pair<const Eigen::MatrixXcd, const std::vector<double>> soc_hamiltonian(const std::vector<int>& twoss, const std::vector<std::vector<double>>& energies, const std::array<Eigen::MatrixXcd, 3>& hsomo, const std::vector<std::vector<Eigen::MatrixXd>>& pdm_dict)
{
    constexpr auto au2cm = 219474.631115585274529;
    // twoss: the 2S list of all states;
    // energies: the energy list of all states, energies[i] is the energy of state i ,j is the jth roots;
    // hsomo : the hamiltonian matrix of spin orbital couplings, with X Y Z;
    // pdm_dict: the pdm dictionary, with key (ibra, jket) and partial density matrix;
    // return: the hamiltonian matrix of spin orbital couplings with complex values;

    assert(twoss.size() == energies.size());

    int nstates = twoss.size(); // the number of states for interaction;
    auto xnroots = std::vector<int>(nstates);
    std::transform(energies.begin(), energies.end(), xnroots.begin(), [](const std::vector<double>& e) {
        return e.size();
    }); // the number of roots for each state;
    int total_roots = std::accumulate(xnroots.begin(), xnroots.end(), 0); // the total number of roots;
    // the total number of roots should be equal to the number of pdm_dict;
    std::vector<double> eners;
    std::vector<int> xtwos;
    eners.reserve(total_roots);
    xtwos.reserve(total_roots);
    // Loop over each state to populate the energy and spin multiplicity arrays
    for (int i = 0; i < nstates; ++i) {
        // Append all energy roots of the current state to the eners vector
        eners.insert(eners.end(), energies[i].begin(), energies[i].end());
        // Append the spin multiplicity (2S) of the current state repeated for each root
        xtwos.insert(xtwos.end(), xnroots[i], twoss[i]);
    }
    // check the size of all matrices;
    auto pdm0 = pdm_dict.at(0).at(0);
    assert(pdm0.rows() == pdm0.cols());
    const int ncas = pdm0.rows();

    Eigen::MatrixXd pdm = Eigen::MatrixXd::Zero(ncas, ncas);

    constexpr double threshold = 29.0; // cm*-1
    int n_mstates = 0;
    for (int i = 0; i < nstates; ++i) {
        n_mstates += (twoss[i] + 1) * xnroots[i];
    }
    // hsiso and hdiag
    Eigen::MatrixXcd hsiso = Eigen::MatrixXcd::Zero(n_mstates, n_mstates);
    Eigen::VectorXcd hdiag = Eigen::VectorXcd::Zero(n_mstates);
    // Loop over each state to calculate the hsiso matrix
    std::vector<std::vector<double>> qls;
    int imb = 0;
    for (int ibra = 0; ibra < total_roots; ++ibra) {
        int imk = 0;
        auto tjb = xtwos[ibra];
        for (int iket = 0; iket < total_roots; ++iket) {
            auto tjk = xtwos[iket];
            if (ibra >= iket) {
                if (std::abs(xtwos[ibra] - xtwos[iket]) > 2) {
                    continue;
                }
                auto pdm = pdm_dict.at(ibra).at(iket);
                for (int ibm = 0; ibm <= tjb; ++ibm) {
                    for (int ikm = 0; ikm <= tjk; ++ikm) {
                        auto xpdm = xyz_proj(spin_projection(pdm, tjb, 2, tjk, ibm, ikm));
                        std::complex<double> somat = std::inner_product(
                            xpdm.begin(), xpdm.end(), hsomo.begin(), std::complex<double>(0.0),
                            std::plus<>(),
                            [](const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
                                return (a.cwiseProduct(b)).sum();
                            });
                        // place the matrix element in the hsiso matrix;
                        hsiso(ibm + imb, ikm + imk) = somat;
                        somat *= au2cm;
                        if (std::abs(somat.real()) > threshold || std::abs(somat.imag()) > threshold) {
                            fmt::println(
                                "I1 = {:4d} (E1 = {:15.8f}) S1 = {:4.1f} MS1 = {:4.1f} "
                                "I2 = {:4d} (E2 = {:15.8f}) S2 = {:4.1f} MS2 = {:4.1f} Re = {: 9.3f} Im = {: 9.3f}",
                                ibra,
                                eners[ibra],
                                static_cast<double>(tjb) / 2.0,
                                static_cast<double>(-tjb) / 2.0 + ibm,
                                iket,
                                eners[iket],
                                static_cast<double>(tjk) / 2.0,
                                static_cast<double>(-tjk) / 2.0 + ikm,
                                somat.real(),
                                somat.imag());
                        }
                    }
                }
                imk += tjk + 1;
            }
            for (int ibm = 0; ibm <= tjb; ++ibm) {
                // qls.push_back({ ibra, eners[ibra], static_cast<double>(tjb) / 2.0, -static_cast<double>(tjb) / 2.0 + ibm });
            }
        }
        for (int diags = imb; diags < imb + tjb + 1; ++diags) {
            hdiag(diags) += eners[ibra];
        }
        imb += tjb + 1;
    }
    for (int i = 0; i < n_mstates; ++i) {
        for (int j = 0; j < n_mstates; ++j) {
            if (i >= j) {
                hsiso(j, i) = std::conj(hsiso(i, j));
            }
        }
    }

    double symm_err = (hsiso - hsiso.adjoint()).cwiseAbs().norm();
    fmt::println("SYMM Error (should be small) = {:15.8f}", symm_err);
    assert(symm_err < 1e-10);
    // add the diagonal elements;
    for (int i = 0; i < n_mstates; ++i) {
        hsiso(i, i) += hdiag(i);
    }
    // std::cout << "SISO:\n"
    //           << hsiso << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(hsiso);

    const Eigen::VectorXd& heig = es.eigenvalues().real();
    const Eigen::MatrixXcd& hvec = es.eigenvectors();

    fmt::println("Total energies including SO-coupling:");
    Eigen::VectorXd xhdiag(heig.size());

    for (int i = 0; i < heig.size(); i++) {
        Eigen::VectorXd shvec = Eigen::VectorXd::Zero(heig.size());
        double ssq = 0.0;
        int imb = 0;

        for (int ibra = 0; ibra < eners.size(); ++ibra) {
            const auto tjb = xtwos[ibra];
            const auto block_size = tjb + 1;
            shvec(ibra) = hvec.block(imb, i, block_size, 1).squaredNorm();
            const double factor = (tjb + 2.0) * tjb / 4.0;
            ssq += shvec(ibra) * factor;
            imb += block_size;
        }

        assert(std::abs(shvec.sum() - 1.0) < 1e-7 && "Probability sum error");

        int iv;
        shvec.maxCoeff(&iv);
        xhdiag(i) = eners[iv];
        fmt::print(
            " State {:4d} Total energy: {:15.8f} <S^2>: {:12.6f} | largest |coeff|**2 {:10.6f} from I = {:4d} E = {:15.8f} S = {:4.1f}\n",
            i, heig[i], ssq, shvec[iv], iv, eners[iv], xtwos[iv] / 2.0);
    }
    std::vector<double> soc_energies(heig.size());
    Eigen::VectorXd::Map(soc_energies.data(), heig.size()) = heig;
    return { hsiso, soc_energies };
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    //  constexpr auto ncas { 27 };
    //  std::vector<int> twoss { 0, 2, 4, 6 };
    constexpr auto ncas { 22 };
    std::vector<int> twoss { 1, 3 };

    std::vector<std::vector<double>> energies;
    for (const auto& spin : twoss) {
        std::vector<double> energy_list;
        std::string filename = fmt::format("ene_{}.txt", spin);
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            int idx;
            double energy;
            if (iss >> idx >> energy) {
                energy_list.push_back(energy);
            }
        }
        energies.push_back(std::move(energy_list));
    }

    assert(energies.size() == twoss.size());

    auto total_roots = std::accumulate(energies.begin(), energies.end(), 0, [](int sum, const std::vector<double>& v) {
        return sum + v.size();
    });

    std::vector<std::vector<Eigen::MatrixXd>> pdms(total_roots, std::vector<Eigen::MatrixXd>(total_roots, Eigen::MatrixXd::Zero(ncas, ncas)));
    std::ifstream pdms_file("pdms.bin", std::ios::binary);
    if (!pdms_file) {
        std::cerr << "Error opening  pdms.bin " << std::endl;
        return 1;
    }
    int32_t entry_count;
    pdms_file.read(reinterpret_cast<char*>(&entry_count), sizeof(int32_t));
    if (entry_count > (total_roots * total_roots)) {
        std::cerr << "Unexpected entry count: " << entry_count << std::endl;
        return 1;
    }
    for (int n = 0; n < entry_count; ++n) {
        int32_t i, j;
        pdms_file.read(reinterpret_cast<char*>(&i), sizeof(int32_t));
        pdms_file.read(reinterpret_cast<char*>(&j), sizeof(int32_t));
        int32_t rows, cols;
        pdms_file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        pdms_file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));

        assert((static_cast<int32_t>(ncas) == rows) && (static_cast<int32_t>(ncas) == cols));

        Eigen::MatrixXd mat(ncas, ncas);
        pdms_file.read(reinterpret_cast<char*>(mat.data()),
            ncas * ncas * sizeof(double));
        mat.transposeInPlace();
        pdms.at(i).at(j) = mat;
        fmt::println("i = {}, j = {}", i, j);
        std::cout << pdms[i][j].format(Format::NumpyFmt) << std::endl;
    }

    // read hso.bin
    std::vector<std::string> labels { "hso_x.bin", "hso_y.bin", "hso_z.bin" };
    std::array<Eigen::MatrixXcd, 3> hsomo;
    for (int i = 0; i < 3; i++) {
        // hsomo[i] = Eigen::MatrixXcd::Zero(ncas, ncas);
        std::ifstream file(labels[i], std::ios::binary);
        if (!file) {
            std::cerr << "Error: Failed to open " + labels[i] << std::endl;
            return 1;
        }
        int32_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        fmt::print("rows = {}, cols = {}\n", rows, cols);
        assert((static_cast<int32_t>(ncas) == rows) && (static_cast<int32_t>(ncas) == cols));
        Eigen::MatrixXcd mat(ncas, ncas);
        file.read(reinterpret_cast<char*>(mat.data()),
            ncas * ncas * sizeof(std::complex<double>));
        mat.transposeInPlace();
        hsomo[i] = mat;

        std::cout << "hsomo[" << i << "] = " << '\n'
                  << hsomo[i].format(Format::NumpyFmt) << std::endl;
    }

    auto [hsiso, soc_e] = soc_hamiltonian(twoss, energies, hsomo, pdms);
    constexpr auto au2ev = 27.21139;
    constexpr auto au2cm = 219474.631115585274529;

    double min_e = *std::min_element(soc_e.begin(), soc_e.end());
    std::ofstream ofs("soc_e.txt");
    if (!ofs.is_open()) {
        throw std::runtime_error(std::string("Failed to open file: ") + "soc_e");
    }

    for (size_t ix = 0; ix < soc_e.size(); ++ix) {
        double ex = soc_e[ix];
        auto format = fmt::format("{:5d} {:20.10f} Ha {:15.6f} eV {:10.4f} cm-1\n",
            static_cast<int>(ix), ex, (ex - min_e) * au2ev, (ex - min_e) * au2cm);
        fmt::print(format);
        ofs << format;
    }

    ofs.close();

    fmt::println("total time = {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start)
            .count());
    return 0;
}