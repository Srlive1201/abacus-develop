//=======================
// AUTHOR : Rong Shi
// DATE :   2022-12-09
//=======================

#ifndef RPA_LRI_HPP
#define RPA_LRI_HPP
#include "RPA_LRI.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include "librpa.h"

template <typename T, typename Tdata> void RPA_LRI<T, Tdata>::init(const MPI_Comm& mpi_comm_in, const K_Vectors& kv_in)
{
    ModuleBase::TITLE("RPA_LRI", "init");
    ModuleBase::timer::tick("RPA_LRI", "init");
    this->mpi_comm = mpi_comm_in;
    this->lcaos = exx_lri_rpa.lcaos;
    this->abfs = exx_lri_rpa.abfs;
    this->abfs_ccp = exx_lri_rpa.abfs_ccp;
    this->p_kv = &kv_in;
    this->cal_rpa_cv();
    //	this->cv = std::move(exx_lri_rpa.cv);
    //    exx_lri_rpa.cv = exx_lri_rpa.cv;
}

template <typename T, typename Tdata> void RPA_LRI<T, Tdata>::cal_rpa_cv()
{
    std::vector<TA> atoms(GlobalC::ucell.nat);
    for (int iat = 0; iat < GlobalC::ucell.nat; ++iat)
        atoms[iat] = iat;
    const std::array<Tcell, Ndim> period = {p_kv->nmp[0], p_kv->nmp[1], p_kv->nmp[2]};

    const std::array<Tcell, Ndim> period_Vs = LRI_CV_Tools::cal_latvec_range<Tcell>(1 + this->info.ccp_rmesh_times);
    const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>> list_As_Vs
        = RI::Distribute_Equally::distribute_atoms(this->mpi_comm, atoms, period_Vs, 2, false);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs
        = exx_lri_rpa.cv.cal_Vs(list_As_Vs.first, list_As_Vs.second[0], {{"writable_Vws", true}});
    this->Vs_period = RI::RI_Tools::cal_period(Vs, period);

    const std::array<Tcell, Ndim> period_Cs = LRI_CV_Tools::cal_latvec_range<Tcell>(2);
    const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>> list_As_Cs
        = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Cs, 2, false);

    std::pair<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>,
              std::array<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>, 3>>
        Cs_dCs = exx_lri_rpa.cv.cal_Cs_dCs(list_As_Cs.first,
                                           list_As_Cs.second[0],
                                           {{"cal_dC", false},
                                            {"writable_Cws", true},
                                            {"writable_dCws", true},
                                            {"writable_Vws", false},
                                            {"writable_dVws", false}});
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> &Cs = std::get<0>(Cs_dCs);
    this->Cs_period = RI::RI_Tools::cal_period(Cs, period);
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::cal_postSCF_exx(const elecstate::DensityMatrix<T, Tdata>& dm,
    const MPI_Comm& mpi_comm_in,
    const K_Vectors& kv)
{
	Mix_DMk_2D mix_DMk_2D;
	mix_DMk_2D.set_nks(kv.get_nks(), GlobalV::GAMMA_ONLY_LOCAL);
	mix_DMk_2D.set_mixing(nullptr);
	mix_DMk_2D.mix(dm.get_DMK_vector(), true);
	const std::vector<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>>
		Ds = GlobalV::GAMMA_ONLY_LOCAL
			? RI_2D_Comm::split_m2D_ktoR<Tdata>(kv, mix_DMk_2D.get_DMk_gamma_out(), *dm.get_paraV_pointer())
			: RI_2D_Comm::split_m2D_ktoR<Tdata>(kv, mix_DMk_2D.get_DMk_k_out(), *dm.get_paraV_pointer());

    // set parameters for bare Coulomb potential
    GlobalC::exx_info.info_global.ccp_type = Conv_Coulomb_Pot_K::Ccp_Type::Hf;
    GlobalC::exx_info.info_global.hybrid_alpha = 1;
    GlobalC::exx_info.info_ri.ccp_rmesh_times = INPUT.rpa_ccp_rmesh_times;

    exx_lri_rpa.init(mpi_comm_in, kv);
    exx_lri_rpa.cal_exx_ions();
    exx_lri_rpa.cal_exx_elec(Ds, *dm.get_paraV_pointer());
    // cout<<"postSCF_Eexx: "<<exx_lri_rpa.Eexx<<endl;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_for_RPA(const Parallel_Orbitals& parav,
    const psi::Psi<T>& psi,
    const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DFT_RPA_interface", "out_for_RPA");
    this->out_bands(pelec);
    this->out_eigen_vector(parav, psi);
    this->out_struc();

    std::cout << "rpa_pca_threshold: " << this->info.pca_threshold << std::endl;
    std::cout << "rpa_ccp_rmesh_times: " << this->info.ccp_rmesh_times << std::endl;
    std::cout << "rpa_lcao_exx(Ha): " << std::fixed << std::setprecision(15) << exx_lri_rpa.Eexx / 2.0 << std::endl;
    this->out_Cs();
    this->out_coulomb_k();

    std::cout << "etxc(Ha): " << std::fixed << std::setprecision(15) << pelec->f_en.etxc / 2.0 << std::endl;
    std::cout << "etot(Ha): " << std::fixed << std::setprecision(15) << pelec->f_en.etot / 2.0 << std::endl;
    std::cout << "Etot_without_rpa(Ha): " << std::fixed << std::setprecision(15)
              << (pelec->f_en.etot - pelec->f_en.etxc + exx_lri_rpa.Eexx) / 2.0 << std::endl;

    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_eigen_vector(const Parallel_Orbitals& parav, const psi::Psi<T>& psi)
{

    ModuleBase::TITLE("DFT_RPA_interface", "out_eigen_vector");

    const int nks_tot = GlobalV::NSPIN == 2 ? p_kv->get_nks() / 2 : p_kv->get_nks();
    const int npsin_tmp = GlobalV::NSPIN == 2 ? 2 : 1;
    const std::complex<double> zero(0.0, 0.0);

    for (int ik = 0; ik < nks_tot; ik++)
    {
        std::stringstream ss;
        ss << "KS_eigenvector_" << ik << ".dat";

        std::ofstream ofs;
        if (GlobalV::MY_RANK == 0)
            ofs.open(ss.str().c_str(), std::ios::out);
        std::vector<ModuleBase::ComplexMatrix> is_wfc_ib_iw(npsin_tmp);
        for (int is = 0; is < npsin_tmp; is++)
        {
            is_wfc_ib_iw[is].create(GlobalV::NBANDS, GlobalV::NLOCAL);
            for (int ib_global = 0; ib_global < GlobalV::NBANDS; ++ib_global)
            {
                std::vector<std::complex<double>> wfc_iks(GlobalV::NLOCAL, zero);

                const int ib_local = parav.global2local_col(ib_global);

                if (ib_local >= 0)
                    for (int ir = 0; ir < psi.get_nbasis(); ir++)
                        wfc_iks[parav.local2global_row(ir)] = psi(ik + nks_tot * is, ib_local, ir);

                std::vector<std::complex<double>> tmp = wfc_iks;
#ifdef __MPI
                MPI_Allreduce(&tmp[0], &wfc_iks[0], GlobalV::NLOCAL, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif
                for (int iw = 0; iw < GlobalV::NLOCAL; iw++)
                    is_wfc_ib_iw[is](ib_global, iw) = wfc_iks[iw];
            } // ib
        } // is
        ofs << ik + 1 << std::endl;
        for (int iw = 0; iw < GlobalV::NLOCAL; iw++)
        {
            for (int ib = 0; ib < GlobalV::NBANDS; ib++)
            {
                for (int is = 0; is < npsin_tmp; is++)
                    ofs << std::setw(21) << std::fixed << std::setprecision(15) << is_wfc_ib_iw[is](ib, iw).real()
                        << std::setw(21) << std::fixed << std::setprecision(15) << is_wfc_ib_iw[is](ib, iw).imag()
                        << std::endl;
            }
        }
        ofs.close();
    } // ik
    return;
}

template <typename T, typename Tdata> void RPA_LRI<T, Tdata>::out_struc()
{
    if (GlobalV::MY_RANK != 0)
        return;
    ModuleBase::TITLE("DFT_RPA_interface", "out_struc");
    double TWOPI_Bohr2A = ModuleBase::TWO_PI * ModuleBase::BOHR_TO_A;
    const int nks_tot = GlobalV::NSPIN == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    ModuleBase::Matrix3 lat = GlobalC::ucell.latvec / ModuleBase::BOHR_TO_A;
    ModuleBase::Matrix3 G = GlobalC::ucell.G * TWOPI_Bohr2A;
    std::stringstream ss;
    ss << "stru_out";
    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);
    ofs << lat.e11 << std::setw(15) << lat.e12 << std::setw(15) << lat.e13 << std::endl;
    ofs << lat.e21 << std::setw(15) << lat.e22 << std::setw(15) << lat.e23 << std::endl;
    ofs << lat.e31 << std::setw(15) << lat.e32 << std::setw(15) << lat.e33 << std::endl;

    ofs << G.e11 << std::setw(15) << G.e12 << std::setw(15) << G.e13 << std::endl;
    ofs << G.e21 << std::setw(15) << G.e22 << std::setw(15) << G.e23 << std::endl;
    ofs << G.e31 << std::setw(15) << G.e32 << std::setw(15) << G.e33 << std::endl;

    ofs << p_kv->nmp[0] << std::setw(6) << p_kv->nmp[1] << std::setw(6) << p_kv->nmp[2]
        << std::setw(6) << std::endl;

    for (int ik = 0; ik != nks_tot; ik++)
        ofs << std::setw(15) << std::fixed << std::setprecision(9) << p_kv->kvec_c[ik].x * TWOPI_Bohr2A
            << std::setw(15) << std::fixed << std::setprecision(9) << p_kv->kvec_c[ik].y * TWOPI_Bohr2A
            << std::setw(15) << std::fixed << std::setprecision(9) << p_kv->kvec_c[ik].z * TWOPI_Bohr2A
            << std::endl;
    ofs.close();
    return;
}

template <typename T, typename Tdata> void RPA_LRI<T, Tdata>::out_bands(const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DFT_RPA_interface", "out_bands");
    if (GlobalV::MY_RANK != 0)
        return;
    const int nks_tot = GlobalV::NSPIN == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    const int nspin_tmp = GlobalV::NSPIN == 2 ? 2 : 1;
    std::stringstream ss;
    ss << "band_out";
    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);
    ofs << nks_tot << std::endl;
    ofs << GlobalV::NSPIN << std::endl;
    ofs << GlobalV::NBANDS << std::endl;
    ofs << GlobalV::NLOCAL << std::endl;
    ofs << (pelec->eferm.ef / 2.0) << std::endl;

    for (int ik = 0; ik != nks_tot; ik++)
    {
        for (int is = 0; is != nspin_tmp; is++)
        {
            ofs << std::setw(6) << ik + 1 << std::setw(6) << is + 1 << std::endl;
            for (int ib = 0; ib != GlobalV::NBANDS; ib++)
                ofs << std::setw(5) << ib + 1 << "   " << std::setw(8) << pelec->wg(ik + is * nks_tot, ib) * nks_tot
                    << std::setw(18) << std::fixed << std::setprecision(8) << pelec->ekb(ik + is * nks_tot, ib) / 2.0
                    << std::setw(18) << std::fixed << std::setprecision(8)
                    << pelec->ekb(ik + is * nks_tot, ib) * ModuleBase::Ry_to_eV << std::endl;
        }
    }
    ofs.close();
    return;
}

template <typename T, typename Tdata> void RPA_LRI<T, Tdata>::out_Cs()
{
    std::stringstream ss;
    ss << "Cs_data_" << GlobalV::MY_RANK << ".txt";
    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);
    ofs << GlobalC::ucell.nat << "    " << 0 << std::endl;
    for (auto &Ip: this->Cs_period)
    {
        size_t I = Ip.first;
        size_t i_num = GlobalC::ucell.atoms[GlobalC::ucell.iat2it[I]].nw;
        for (auto &JPp: Ip.second)
        {
            size_t J = JPp.first.first;
            auto R = JPp.first.second;
            auto &tmp_Cs = JPp.second;
            size_t j_num = GlobalC::ucell.atoms[GlobalC::ucell.iat2it[J]].nw;

            ofs << I + 1 << "   " << J + 1 << "   " << R[0] << "   " << R[1] << "   " << R[2] << "   " << i_num
                << std::endl;
            ofs << j_num << "   " << tmp_Cs.shape[0] << std::endl;
            for (int i = 0; i != i_num; i++)
                for (int j = 0; j != j_num; j++)
                    for (int mu = 0; mu != tmp_Cs.shape[0]; mu++)
                        ofs << std::setw(15) << std::fixed << std::setprecision(9) << tmp_Cs(mu, i, j) << std::endl;
        }
    }
    ofs.close();
    return;
}

template <typename T, typename Tdata> void RPA_LRI<T, Tdata>::out_coulomb_k()
{
    int all_mu = 0;
    vector<int> mu_shift(GlobalC::ucell.nat);
    for (int I = 0; I != GlobalC::ucell.nat; I++)
    {
        mu_shift[I] = all_mu;
        all_mu += exx_lri_rpa.cv.get_index_abfs_size(GlobalC::ucell.iat2it[I]);
    }
    const int nks_tot = GlobalV::NSPIN == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    std::stringstream ss;
    ss << "coulomb_mat_" << GlobalV::MY_RANK << ".txt";

    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);

    ofs << nks_tot << std::endl;
    for (auto &Ip: this->Vs_period)
    {
        auto I = Ip.first;
        size_t mu_num = exx_lri_rpa.cv.get_index_abfs_size(GlobalC::ucell.iat2it[I]);

        for (int ik = 0; ik != nks_tot; ik++)
        {
            std::map<size_t, RI::Tensor<std::complex<double>>> Vq_k_IJ;
            for (auto &JPp: Ip.second)
            {
                auto J = JPp.first.first;

                auto R = JPp.first.second;
                if (J < I)
                    continue;
                RI::Tensor<std::complex<double>> tmp_VR = RI::Global_Func::convert<std::complex<double>>(JPp.second);

                const double arg = 1
                                   * (p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(R) * GlobalC::ucell.latvec))
                                   * ModuleBase::TWO_PI; // latvec
                const std::complex<double> kphase = std::complex<double>(cos(arg), sin(arg));
                if (Vq_k_IJ[J].empty())
                    Vq_k_IJ[J] = RI::Tensor<std::complex<double>>({tmp_VR.shape[0], tmp_VR.shape[1]});
                Vq_k_IJ[J] = Vq_k_IJ[J] + tmp_VR * kphase;
            }
            for (auto &vq_Jp: Vq_k_IJ)
            {
                auto iJ = vq_Jp.first;
                auto &vq_J = vq_Jp.second;
                size_t nu_num = exx_lri_rpa.cv.get_index_abfs_size(GlobalC::ucell.iat2it[iJ]);
                ofs << all_mu << "   " << mu_shift[I] + 1 << "   " << mu_shift[I] + mu_num << "  " << mu_shift[iJ] + 1
                    << "   " << mu_shift[iJ] + nu_num << std::endl;
                ofs << ik + 1 << "  " << p_kv->wk[ik] / 2.0 * GlobalV::NSPIN << std::endl;
                for (int i = 0; i != vq_J.data->size(); i++)
                {
                    ofs << std::setw(21) << std::fixed << std::setprecision(12) << (*vq_J.data)[i].real()
                        << std::setw(21) << std::fixed << std::setprecision(12) << (*vq_J.data)[i].imag() << std::endl;
                }
            }
        }
    }
    ofs.close();
}

// template<typename Tdata>
// void RPA_LRI<T, Tdata>::init(const MPI_Comm &mpi_comm_in)
// {
// 	if(this->info == this->exx.info)
// 	{
// 		this->lcaos = this->exx.lcaos;
// 		this->abfs = this->exx.abfs;
// 		this->abfs_ccp = this->exx.abfs_ccp;

// 		exx_lri_rpa.cv = std::move(this->exx.cv);
// 	}
// 	else
// 	{
// 		this->lcaos = ...
// 		this->abfs = ...
// 		this->abfs_ccp = ...

// 		exx_lri_rpa.cv.set_orbitals(
// 			this->lcaos, this->abfs, this->abfs_ccp,
// 			this->info.kmesh_times, this->info.ccp_rmesh_times );
// 	}

// //	for( size_t T=0; T!=this->abfs.size(); ++T )
// //		GlobalC::exx_info.info_ri.abfs_Lmax = std::max( GlobalC::exx_info.info_ri.abfs_Lmax,
// static_cast<int>(this->abfs[T].size())-1 );

// }

// template<typename Tdata>
// void RPA_LRI<T, Tdata>::cal_rpa_ions()
// {
// 	// this->rpa_lri.set_parallel(this->mpi_comm, atoms_pos, latvec, period);

// 	if(this->info == this->exx.info)
// 		exx_lri_rpa.cv.Vws = std::move(this->exx.cv.Vws);

// 	const std::array<Tcell,Ndim> period_Vs = LRI_CV_Tools::cal_latvec_range<Tcell>(1+this->info.ccp_rmesh_times);
// 	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
// 		list_As_Vs = RI::Distribute_Equally::distribute_atoms(this->mpi_comm, atoms, period_Vs, 2, false);

// 	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
// 		Vs = exx_lri_rpa.cv.cal_Vs(
// 			list_As_Vs.first, list_As_Vs.second[0],
// 			{{"writable_Vws",true}});

// 	// Vs[iat0][{iat1,cell1}]	按 (iat0,iat1) 分进程，每个进程有所有 cell1
// 	Vqs = FFT(Vs);
// 	out_Vs(Vqs);

// 	if(this->info == this->exx.info)
// 		exx_lri_rpa.cv.Cws = std::move(this->exx.cv.Cws);

// 	const std::array<Tcell,Ndim> period_Cs = LRI_CV_Tools::cal_latvec_range<Tcell>(2);
// 	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
// 		list_As_Cs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Cs, 2, false);

// 	std::pair<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>, std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>>
// 		Cs_dCs = exx_lri_rpa.cv.cal_Cs_dCs(
// 			list_As_Cs.first, list_As_Cs.second[0],
// 			{{"cal_dC",false},
// 			 {"writable_Cws",true}, {"writable_dCws",true}, {"writable_Vws",false}, {"writable_dVws",false}});
// 	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> &Cs = std::get<0>(Cs_dCs);

// 	out_Cs(Cs);

// 	// rpa_lri.set_Cs(Cs);
// }

template <typename T, typename Tdata>
void RPA_LRI<T,Tdata>::tran_data_to_librpa(const Parallel_Orbitals &parav,
                                 const psi::Psi<T>& psi,
                                 const elecstate::ElecState *pelec)
{
    ModuleBase::TITLE("DFT_RPA_interface", "tran_data_to_librpa");
    std::cout << "rpa_pca_threshold: " << this->info.pca_threshold << std::endl;
    std::cout << "rpa_ccp_rmesh_times: " << this->info.ccp_rmesh_times << std::endl;
    std::cout << "rpa_lcao_exx(Ha): " << std::fixed << std::setprecision(15) << exx_lri_rpa.Eexx / 2.0 << std::endl;
    std::cout << "etxc(Ha): " << std::fixed << std::setprecision(15) << pelec->f_en.etxc / 2.0 << std::endl;
    std::cout << "etot(Ha): " << std::fixed << std::setprecision(15) << pelec->f_en.etot / 2.0 << std::endl;
    std::cout << "Etot_without_rpa(Ha): " << std::fixed << std::setprecision(15)
              << (pelec->f_en.etot - pelec->f_en.etxc + exx_lri_rpa.Eexx) / 2.0 << std::endl;
    printf("IN abacus myid : %d\n",GlobalV::MY_RANK);
    initialize_librpa_environment(this->mpi_comm,0,1,"librpa.out");
    this->tran_bands(pelec);
    this->tran_eigen_vector(parav, psi); 
    this->tran_struc();
    this->tran_Cs();
    this->tran_coulomb_k();

    LibRPAParams rpa_params;
    strcpy(rpa_params.task, "rpa");
    strcpy(rpa_params.output_file, "stdout");
    strcpy(rpa_params.output_dir, "librpa.d");
    rpa_params.nfreq = 12;
    set_librpa_params(&rpa_params);
    printf("Before LibRPA myid : %d\n",GlobalV::MY_RANK);
    MPI_Barrier(MPI_COMM_WORLD);
    run_librpa_main();
    finalize_librpa_environment();
}

template <typename T, typename Tdata> 
void RPA_LRI<T, Tdata>::tran_bands(const elecstate::ElecState *pelec)
{
    ModuleBase::TITLE("DFT_RPA_interface", "tran_bands");
    // if (GlobalV::MY_RANK != 0)
    //     return;
    const int nks_tot = GlobalV::NSPIN == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    const int nspin_tmp = GlobalV::NSPIN == 2 ? 2 : 1;
    set_dimension(GlobalV::NSPIN,nks_tot,GlobalV::NBANDS,GlobalV::NLOCAL,GlobalC::ucell.nat);
    ModuleBase::matrix ekb_Ha= pelec->ekb* 0.5;
    ModuleBase::matrix wg_nks= pelec->wg*  nks_tot;
    set_wg_ekb_efermi(GlobalV::NSPIN,nks_tot,GlobalV::NBANDS,wg_nks.c, ekb_Ha.c, pelec->eferm.ef / 2.0 );
    printf("tran_band myid : %d\n",GlobalV::MY_RANK);
    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::tran_eigen_vector(const Parallel_Orbitals &parav, const psi::Psi<T>& psi)
{

    ModuleBase::TITLE("DFT_RPA_interface", "out_eigen_vector");

    const int nks_tot = GlobalV::NSPIN == 2 ? p_kv->get_nks() / 2 : p_kv->get_nks();
    const int npsin_tmp = GlobalV::NSPIN == 2 ? 2 : 1;
    const std::complex<double> zero(0.0, 0.0);

    for (int ik = 0; ik < nks_tot; ik++)
    {
        std::vector<ModuleBase::ComplexMatrix> is_wfc_ib_iw(npsin_tmp);
        for (int is = 0; is < npsin_tmp; is++)
        {
            is_wfc_ib_iw[is].create(GlobalV::NBANDS, GlobalV::NLOCAL);
            for (int ib_global = 0; ib_global < GlobalV::NBANDS; ++ib_global)
            {
                std::vector<std::complex<double>> wfc_iks(GlobalV::NLOCAL, zero);

                const int ib_local = parav.global2local_col(ib_global);

                if (ib_local >= 0)
                    for (int ir = 0; ir < psi.get_nbasis(); ir++)
                        wfc_iks[parav.local2global_row(ir)] = psi(ik + nks_tot * is, ib_local, ir);

                std::vector<std::complex<double>> tmp = wfc_iks;
#ifdef __MPI
                MPI_Allreduce(&tmp[0], &wfc_iks[0], GlobalV::NLOCAL, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif
                for (int iw = 0; iw < GlobalV::NLOCAL; iw++)
                    is_wfc_ib_iw[is](ib_global, iw) = wfc_iks[iw];
            } // ib
            ModuleBase::matrix wfc_real=is_wfc_ib_iw[is].real();
            ModuleBase::matrix wfc_imag(GlobalV::NBANDS,GlobalV::NLOCAL);
            for(int i=0;i!=is_wfc_ib_iw[is].size;i++)
                wfc_imag.c[i]=is_wfc_ib_iw[is].c[i].imag();
            
            set_ao_basis_wfc(is,ik, wfc_real.c, wfc_imag.c);
        } // is
    } // ik
    printf("tran_eigenvector : %d\n",GlobalV::MY_RANK);
    return;
}

template <typename T, typename Tdata> 
void RPA_LRI<T, Tdata>::tran_struc()
{
    // if (GlobalV::MY_RANK != 0)
    //     return;
    ModuleBase::TITLE("DFT_RPA_interface", "out_struc");
    double TWOPI_Bohr2A = ModuleBase::TWO_PI * ModuleBase::BOHR_TO_A;
    const int nks_tot = GlobalV::NSPIN == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    ModuleBase::Matrix3 lat = GlobalC::ucell.latvec / ModuleBase::BOHR_TO_A;
    ModuleBase::Matrix3 G = GlobalC::ucell.G * TWOPI_Bohr2A;
   
    ModuleBase::matrix lat_mat=lat.to_matrix();
    ModuleBase::matrix G_mat=G.to_matrix();
    
    set_latvec_and_G( lat_mat.c, G_mat.c);
    
    ModuleBase::matrix kvec_mat(nks_tot,3);
    vector<int> irk_list;
    vector<double> irk_weight;
    for (int ik = 0; ik != nks_tot; ik++)
    {
        // need to change the irk_list to real ibz_index!
        irk_list.push_back(ik);
        irk_weight.push_back(1.0/nks_tot);
        kvec_mat(ik,0)=p_kv->kvec_c[ik].x* TWOPI_Bohr2A;
        kvec_mat(ik,1)=p_kv->kvec_c[ik].y* TWOPI_Bohr2A;
        kvec_mat(ik,2)=p_kv->kvec_c[ik].z* TWOPI_Bohr2A;
    }
    // for(int i=0 ;i!=kvec_mat.nr* kvec_mat.nc;i++)
    //     std::cout<<kvec_mat.c[i]<<"   ";
    //std::cout<<"abacus nkstot_ibz"<<p_kv->nkstot_ibz<<std::endl;
    set_kgrids_kvec_tot(p_kv->nmp[0],p_kv->nmp[1],p_kv->nmp[2], kvec_mat.c);
    set_ibz2bz_index_and_weight(nks_tot,irk_list.data(),irk_weight.data());
    printf("tran_stru myid : %d\n",GlobalV::MY_RANK);
    return;
}

template <typename T, typename Tdata> 
void RPA_LRI<T, Tdata>::tran_Cs()
{
    
    for (auto &Ip: this->Cs_period)
    {
        size_t I = Ip.first;
        size_t i_num = GlobalC::ucell.atoms[GlobalC::ucell.iat2it[I]].nw;
        for (auto &JPp: Ip.second)
        {
            size_t J = JPp.first.first;
            auto R = JPp.first.second;
            auto &tmp_Cs = JPp.second;
            size_t j_num = GlobalC::ucell.atoms[GlobalC::ucell.iat2it[J]].nw;

#ifdef __MKL_RI
			const RI::Tensor<Tdata> Cs_sub = RI::Blas_Interface::omatcopy(
				'T', Tdata{1.0},
				tmp_Cs.reshape({tmp_Cs.shape[0],tmp_Cs.shape[1]*tmp_Cs.shape[2]}));
#else
            std::size_t Ndim12=tmp_Cs.shape[1]*tmp_Cs.shape[2];
			RI::Tensor<Tdata> Cs_sub({Ndim12,tmp_Cs.shape[0]});

			std::vector<Tdata*> Cs_sub_ptr(Ndim12);
			for(int i12=0; i12<Ndim12; ++i12)
				Cs_sub_ptr[i12] = Cs_sub.ptr()+i12*Cs_sub.shape[1]-1;

			const Tdata* Cs_ptr = tmp_Cs.ptr()-1;
			for(std::size_t i0=0; i0<Cs_sub.shape[1]; ++i0)
				for(std::size_t i12=0; i12<Ndim12; ++i12)
					*(++Cs_sub_ptr[i12]) = *(++Cs_ptr);
#endif
            std::cout<<"In tran_Cs: I J: "<<I<<J<<std::endl;
            set_ao_basis_aux(I, J, i_num, j_num, tmp_Cs.shape[0], R.data(), Cs_sub.ptr(),0);
            std::cout<<"end set_ao_basis_aux!"<<std::endl;
        }
    }
    printf("tran_Cs myid : %d\n",GlobalV::MY_RANK);
    return;
}

template <typename T, typename Tdata> 
void RPA_LRI<T, Tdata>::tran_coulomb_k()
{
    int all_mu = 0;
    vector<int> mu_shift(GlobalC::ucell.nat);
    for (int I = 0; I != GlobalC::ucell.nat; I++)
    {
        mu_shift[I] = all_mu;
        all_mu += exx_lri_rpa.cv.get_index_abfs_size(GlobalC::ucell.iat2it[I]);
    }
    const int nks_tot = GlobalV::NSPIN == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    
    for (auto &Ip: this->Vs_period)
    {
        auto I = Ip.first;
        size_t mu_num = exx_lri_rpa.cv.get_index_abfs_size(GlobalC::ucell.iat2it[I]);

        for (int ik = 0; ik != nks_tot; ik++)
        {
            std::map<size_t, RI::Tensor<std::complex<double>>> Vq_k_IJ;
            for (auto &JPp: Ip.second)
            {
                auto J = JPp.first.first;

                auto R = JPp.first.second;
                if (J < I)
                    continue;
                RI::Tensor<std::complex<double>> tmp_VR = RI::Global_Func::convert<std::complex<double>>(JPp.second);

                const double arg = 1
                                   * (p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(R) * GlobalC::ucell.latvec))
                                   * ModuleBase::TWO_PI; // latvec
                const std::complex<double> kphase = std::complex<double>(cos(arg), sin(arg));
                if (Vq_k_IJ[J].empty())
                    Vq_k_IJ[J] = RI::Tensor<std::complex<double>>({tmp_VR.shape[0], tmp_VR.shape[1]});
                Vq_k_IJ[J] = Vq_k_IJ[J] + tmp_VR * kphase;
            }
            for (auto &vq_Jp: Vq_k_IJ)
            {
                auto iJ = vq_Jp.first;
                auto &vq_J = vq_Jp.second;
                size_t nu_num = exx_lri_rpa.cv.get_index_abfs_size(GlobalC::ucell.iat2it[iJ]);
                std::vector<double> vq_real(vq_J.data->size());
                std::vector<double> vq_imag(vq_J.data->size());
                //std::cout<<"  In tran_Vq: mu, nu, vq_size:  "<<mu_num<<"  "<<nu_num<<"  "<<vq_J.data->size()<<std::endl;
                for (int i = 0; i != vq_J.data->size(); i++)
                {
                    vq_real[i] = (*vq_J.data)[i].real();
                    vq_imag[i] = (*vq_J.data)[i].imag();
                }
                set_aux_bare_coulomb_k_atom_pair(ik, I, iJ, mu_num, nu_num,  vq_real.data(),  vq_imag.data());
                
                
                // ofs << all_mu << "   " << mu_shift[I] + 1 << "   " << mu_shift[I] + mu_num << "  " << mu_shift[iJ] + 1
                //     << "   " << mu_shift[iJ] + nu_num << std::endl;
                // ofs << ik + 1 << "  " << p_kv->wk[ik] / 2.0 * GlobalV::NSPIN << std::endl;
                // for (int i = 0; i != vq_J.data->size(); i++)
                // {
                //     ofs << std::setw(21) << std::fixed << std::setprecision(12) << (*vq_J.data)[i].real()
                //         << std::setw(21) << std::fixed << std::setprecision(12) << (*vq_J.data)[i].imag() << std::endl;
                // }
            }
        }
    }
    printf("tran_Vq myid : %d\n",GlobalV::MY_RANK);
}


#endif