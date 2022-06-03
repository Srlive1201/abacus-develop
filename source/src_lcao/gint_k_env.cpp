#include "gint_k.h"
#include "grid_technique.h"
#include "../module_orbital/ORB_read.h"
#include "../src_pw/global.h"
#include "../module_base/ylm.h"
#include "../module_base/timer.h"

void Gint_k::cal_env_k(int ik, const std::complex<double>* wfc_k, double* rho)
{
    ModuleBase::TITLE("Gint_k", "cal_env_k");
    ModuleBase::timer::tick("Gint_k", "cal_env_k");

    // it's a uniform grid to save orbital values, so the delta_r is a constant.
    const double delta_r = GlobalC::ORB.dr_uniform;
    const Numerical_Orbital_Lm* pointer;

    double*** dr; // vectors between atom and grid: [bxyz, maxsize, 3]
    double** distance; // distance between atom and grid: [bxyz, maxsize]
    double*** psir_ylm;
    bool** cal_flag;

    const int max_size = GlobalC::GridT.max_atom;
    if (max_size != 0)
    {
        dr = new double** [GlobalC::pw.bxyz];
        distance = new double* [GlobalC::pw.bxyz];
        psir_ylm = new double** [GlobalC::pw.bxyz];
        cal_flag = new bool* [GlobalC::pw.bxyz];

        for (int i = 0; i < GlobalC::pw.bxyz; i++)
        {
            dr[i] = new double* [max_size];
            distance[i] = new double[max_size];
            psir_ylm[i] = new double* [max_size];
            cal_flag[i] = new bool[max_size];

            ModuleBase::GlobalFunc::ZEROS(distance[i], max_size);
            ModuleBase::GlobalFunc::ZEROS(cal_flag[i], max_size);

            for (int j = 0; j < max_size; j++)
            {
                dr[i][j] = new double[3];
                psir_ylm[i][j] = new double[GlobalC::ucell.nwmax];
                ModuleBase::GlobalFunc::ZEROS(dr[i][j], 3);
                ModuleBase::GlobalFunc::ZEROS(psir_ylm[i][j], GlobalC::ucell.nwmax);
            }
        }
    }

    double mt[3] = { 0,0,0 };
    double* vldr3 = new double[GlobalC::pw.bxyz];
    double v1 = 0.0;
    int* vindex = new int[GlobalC::pw.bxyz];
    ModuleBase::GlobalFunc::ZEROS(vldr3, GlobalC::pw.bxyz);
    ModuleBase::GlobalFunc::ZEROS(vindex, GlobalC::pw.bxyz);
    double phi = 0.0;

    const int nbx = GlobalC::GridT.nbx;
    const int nby = GlobalC::GridT.nby;
    const int nbz_start = GlobalC::GridT.nbzp_start;
    const int nbz = GlobalC::GridT.nbzp;

    for (int i = 0; i < nbx; i++)
    {
        for (int j = 0; j < nby; j++)
        {
            for (int k = nbz_start; k < nbz_start + nbz; k++) // FFT grid
            {
                int grid_index = (k - nbz_start) + j * nbz + i * nby * nbz;

                // get the value: how many atoms has orbital value on this grid.
                const int size = GlobalC::GridT.how_many_atoms[grid_index];
                if (size == 0) continue;

                // (1) initialized the phi * Ylm.
                for (int id = 0; id < size; id++)
                {
                    // there are two parameters we want to know here:

                    // (1) in which bigcell of the meshball the atom in?
                    // value-having-atom's index in this proc
                    const int mcell_index = GlobalC::GridT.bcell_start[grid_index] + id;
                    // the bigcell index of this atom in it's meshball 
                    const int imcell = GlobalC::GridT.which_bigcell[mcell_index];

                    int iat = GlobalC::GridT.which_atom[mcell_index];

                    const int it = GlobalC::ucell.iat2it[iat];
                    const int ia = GlobalC::ucell.iat2ia[iat];

                    //(2) what's the cartesian coordinate of the bigcell?
                    // meshball_positions should be the bigcell position in meshball
                    // to the center of meshball.
                    // calculated in cartesian coordinates
                    // the std::vector from the grid which is now being operated to the atom position.
                    // in meshball language, is the std::vector from imcell to the center cel, plus
                    // tau_in_bigcell.
                    mt[0] = GlobalC::GridT.meshball_positions[imcell][0] - GlobalC::GridT.tau_in_bigcell[iat][0];
                    mt[1] = GlobalC::GridT.meshball_positions[imcell][1] - GlobalC::GridT.tau_in_bigcell[iat][1];
                    mt[2] = GlobalC::GridT.meshball_positions[imcell][2] - GlobalC::GridT.tau_in_bigcell[iat][2];

                    for (int ib = 0; ib < GlobalC::pw.bxyz; ib++)
                    {
                        // meshcell_pos: z is the fastest
                        dr[ib][id][0] = GlobalC::GridT.meshcell_pos[ib][0] + mt[0];
                        dr[ib][id][1] = GlobalC::GridT.meshcell_pos[ib][1] + mt[1];
                        dr[ib][id][2] = GlobalC::GridT.meshcell_pos[ib][2] + mt[2];

                        //find R by tau (for double-check)
                        //dr=r-\tau-R  =>  R= r-\tau-dr
                        //now the eight in one bigcell are same but different with R form index2ucell.
                        // const double Rx1 = (double)i * (GlobalC::ucell.a1[0] + GlobalC::ucell.a2[0] + GlobalC::ucell.a3[0])
                        //     * GlobalC::ucell.lat0 /  (double)GlobalC::pw.nbx    //r
                        //     + GlobalC::GridT.meshcell_pos[ib][0]
                        //     - GlobalC::ucell.atoms[it].tau[ia].x * GlobalC::ucell.lat0
                        //     - dr[ib][id][0];
                        // const double Ry1 = (double)j * (GlobalC::ucell.a1[1] + GlobalC::ucell.a2[1] + GlobalC::ucell.a3[1])
                        //     * GlobalC::ucell.lat0 /  (double)GlobalC::pw.nby
                        //     + GlobalC::GridT.meshcell_pos[ib][1]
                        //     - GlobalC::ucell.atoms[it].tau[ia].y * GlobalC::ucell.lat0
                        //     - dr[ib][id][1];
                        // const double Rz1 = (double)k * (GlobalC::ucell.a1[2] + GlobalC::ucell.a2[2] + GlobalC::ucell.a3[2])
                        //     * GlobalC::ucell.lat0 /  (double)GlobalC::pw.nbz
                        //     + GlobalC::GridT.meshcell_pos[ib][2]
                        //     - GlobalC::ucell.atoms[it].tau[ia].z * GlobalC::ucell.lat0
                        //     - dr[ib][id][2];
                        // std::cout << "R1:" << Rx1 << " " << Ry1 << " " << Rz1 << std::endl;

                        distance[ib][id] = std::sqrt(dr[ib][id][0] * dr[ib][id][0]
                            + dr[ib][id][1] * dr[ib][id][1]
                            + dr[ib][id][2] * dr[ib][id][2]);

                        if (distance[ib][id] <= GlobalC::ORB.Phi[it].getRcut())
                        {
                            cal_flag[ib][id] = true;
                        }
                        else
                        {
                            cal_flag[ib][id] = false;
                            continue;
                        }

                        std::vector<double> ylma;
                        //if(distance[id] > GlobalC::GridT.orbital_rmax) continue;
                        if (distance[ib][id] < 1.0E-9) distance[ib][id] += 1.0E-9;

                        ModuleBase::Ylm::sph_harm(GlobalC::ucell.atoms[it].nwl,
                            dr[ib][id][0] / distance[ib][id],
                            dr[ib][id][1] / distance[ib][id],
                            dr[ib][id][2] / distance[ib][id],
                            ylma);
                        // these parameters are about interpolation
                        // because once we know the distance from atom to grid point,
                        // we can get the parameters we need to do interpolation and
                        // store them first!! these can save a lot of effort.
                        const double position = distance[ib][id] / delta_r;
                        /*
                           this->iq[id] = static_cast<int>(position);
                           this->x0[id] = position - static_cast<double>(iq[id]);
                           this->x1[id] = 1.0 - x0[id];
                           this->x2[id] = 2.0 - x0[id];
                           this->x3[id] = 3.0 - x0[id];
                           this->x12[id] = x1[id]*x2[id] / 6.0;
                           this->x03[id] = x0[id]*x3[id] / 2.0;
                         */
                        int ip;
                        double dx, dx2, dx3;
                        double c1, c2, c3, c4;

                        ip = static_cast<int>(position);
                        dx = position - ip;
                        dx2 = dx * dx;
                        dx3 = dx2 * dx;

                        c3 = 3.0 * dx2 - 2.0 * dx3;
                        c1 = 1.0 - c3;
                        c2 = (dx - 2.0 * dx2 + dx3) * delta_r;
                        c4 = (dx3 - dx2) * delta_r;

                        //		int ip = this->iq[id];
                        //		double A = ip+1.0-position/delta_r;
                        //		double B = 1.0-A;
                        //		double coef1 = (A*A*A-A)/6.0*delta_r*delta_r;
                        //		double coef2 = (B*B*B-B)/6.0*delta_r*delta_r;

                        Atom* atom1 = &GlobalC::ucell.atoms[it];
                        for (int iw = 0; iw < atom1->nw; iw++)
                        {
                            if (atom1->iw2_new[iw])
                            {
                                pointer = &GlobalC::ORB.Phi[it].PhiLN(
                                    atom1->iw2l[iw],
                                    atom1->iw2n[iw]);
                                phi = c1 * pointer->psi_uniform[ip] + c2 * pointer->dpsi_uniform[ip]
                                    + c3 * pointer->psi_uniform[ip + 1] + c4 * pointer->dpsi_uniform[ip + 1];
                            }
                            psir_ylm[ib][id][iw] = phi * ylma[atom1->iw2_ylm[iw]];
                        }
                    }// end ib
                }// end id


                int bindex = 0;
                // z is the fastest, 
                for (int ii = 0; ii < GlobalC::pw.bx; ii++)
                {
                    const int ipart = (i * GlobalC::pw.bx + ii) * GlobalC::rhopw->ny * GlobalC::rhopw->nplane;
                    for (int jj = 0; jj < GlobalC::pw.by; jj++)
                    {
                        const int jpart = (j * GlobalC::pw.by + jj) * GlobalC::rhopw->nplane;
                        for (int kk = 0; kk < GlobalC::pw.bz; kk++)
                        {
                            vindex[bindex] = (k * GlobalC::pw.bz + kk - GlobalC::pw.nczp_start) + jpart + ipart;
                            //	assert(vindex[bindex] < GlobalC::rhopw->nrxx);
                            ++bindex;
                        }
                    }
                }

                for (int ia1 = 0; ia1 < size; ia1++)
                {
                    const int mcell_index1 = GlobalC::GridT.bcell_start[grid_index] + ia1;
                    const int iat = GlobalC::GridT.which_atom[mcell_index1];
                    const int T1 = GlobalC::ucell.iat2it[iat];
                    Atom* atom1 = &GlobalC::ucell.atoms[T1];
                    const int I1 = GlobalC::ucell.iat2ia[iat];

                    //find R by which_unitcell and cal kphase
                    const int id_ucell = GlobalC::GridT.which_unitcell[mcell_index1];
                    const int Rx = GlobalC::GridT.ucell_index2x[id_ucell] + GlobalC::GridT.minu1;
                    const int Ry = GlobalC::GridT.ucell_index2y[id_ucell] + GlobalC::GridT.minu2;
                    const int Rz = GlobalC::GridT.ucell_index2z[id_ucell] + GlobalC::GridT.minu3;
                    ModuleBase::Vector3<double> R((double)Rx, (double)Ry, (double)Rz);
                    //std::cout << "kvec_d: " << GlobalC::kv.kvec_d[ik].x << " " << GlobalC::kv.kvec_d[ik].y << " " << GlobalC::kv.kvec_d[ik].z << std::endl;
                    //std::cout << "kvec_c: " << GlobalC::kv.kvec_c[ik].x << " " << GlobalC::kv.kvec_c[ik].y << " " << GlobalC::kv.kvec_c[ik].z << std::endl;
                    //std::cout << "R: " << R.x << " " << R.y << " " << R.z << std::endl;
                    const double arg = (GlobalC::kv.kvec_d[ik] * R) * ModuleBase::TWO_PI;
                    const double arg1 = (GlobalC::kv.kvec_c[ik] * (R.x * GlobalC::ucell.a1 + R.y * GlobalC::ucell.a2 + R.z * GlobalC::ucell.a3)) * ModuleBase::TWO_PI;
                    //std::cout << "arg0=" << arg << ", arg1=" << arg1 << std::endl;
                    const std::complex<double> kphase = std::complex <double>(cos(arg), sin(arg));

                    // get the start index of local orbitals.
                    const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);
                    for (int ib = 0; ib < GlobalC::pw.bxyz; ib++)
                    {
                        if (cal_flag[ib][ia1])
                        {
                            int iw1_lo;
                            double* psi1 = psir_ylm[ib][ia1];
                            std::complex<double> tmp = (0.0, 0.0);
                            if (GlobalV::NSPIN == 4) // is it a simple add of 2 spins?
                            {
                                for (int is = 0;is < 2;++is)
                                {
                                    iw1_lo = GlobalC::GridT.trace_lo[start1] / GlobalV::NPOL + GlobalC::GridT.lgd / GlobalV::NPOL * is;
                                    for (int iw = 0;iw < atom1->nw;++iw, ++iw1_lo)
                                        tmp += std::complex<double>(psi1[iw], 0.0) * wfc_k[iw1_lo] * kphase;
                                }
                            }
                            else
                            {
                                iw1_lo = GlobalC::GridT.trace_lo[start1];
                                for (int iw = 0; iw < atom1->nw; ++iw, ++iw1_lo)
                                    tmp += std::complex<double>(psi1[iw], 0.0) * wfc_k[iw1_lo] * kphase;
                            }
                            rho[vindex[ib]] += tmp.real();

                            // std::cout << "tmp.real()=" << tmp.real() << std::endl;  //for test
                            // std::cout << "tmp.imag()=" << tmp.imag() << std::endl;  //for test
                        }// cal_flag
                        //std::cout << std::endl;
                    }//ib
                }// ia1

            }// k
        }// j
    }// i

    delete[] vindex;
    delete[] vldr3;

    if (max_size != 0)
    {
        for (int i = 0; i < GlobalC::pw.bxyz; i++)
        {
            for (int j = 0; j < max_size; j++)
            {
                delete[] dr[i][j];
                delete[] psir_ylm[i][j];
            }
            delete[] dr[i];
            delete[] distance[i];
            delete[] psir_ylm[i];
            delete[] cal_flag[i];
        }
        delete[] dr;
        delete[] distance;
        delete[] psir_ylm;
        delete[] cal_flag;
    }
    ModuleBase::timer::tick("Gint_k", "cal_env_k");
    return;
}



