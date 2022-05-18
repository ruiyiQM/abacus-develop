#include "dipole.h"
#include "../module_base/constants.h"
#include "../module_base/timer.h"
#include "../module_base/global_variable.h"
#include "../src_parallel/parallel_reduce.h"

double Dipole::etotefield = 0.0;
double Dipole::tot_dipole = 0.0;
int Dipole::edir = 2;
double Dipole::emaxpos = 0.5;
double Dipole::eopreg = 0.1;
double Dipole::eamp = 0.0;
bool Dipole::first = true;
double Dipole::bvec[3];
double Dipole::bmod;

Dipole::Dipole(){}

Dipole::~Dipole(){}

//=======================================================
// calculate dipole potential in surface calculations
//=======================================================
ModuleBase::matrix Dipole::add_efield(const UnitCell &cell, 
                                        PW_Basis &pwb, 
                                        const int &nspin, 
                                        const double *const *const rho)
{
    ModuleBase::TITLE("Dipole", "add_efield");
    ModuleBase::timer::tick("Dipole", "add_efield");

    ModuleBase::matrix v(nspin, pwb.nrxx);

    // efield only needs to be added on the first iteration, if dipfield
    // is not used. note that for relax calculations it has to be added
    // again on subsequent relax steps.
    if(!GlobalV::DIPOLE && !first)
    {
        return v;
    }
    first = false;

    double latvec;    // latvec along the efield direction
    if(edir == 0)
    {
        bvec[0] = cell.G.e11;
        bvec[1] = cell.G.e12; 
        bvec[2] = cell.G.e13; 
        latvec = cell.a1.norm();
    }
    else if(edir == 1)
    {
        bvec[0] = cell.G.e21;
        bvec[1] = cell.G.e22; 
        bvec[2] = cell.G.e23; 
        latvec = cell.a2.norm();
    }
    else if(edir = 2)
    {
        bvec[0] = cell.G.e31;
        bvec[1] = cell.G.e32; 
        bvec[2] = cell.G.e33; 
        latvec = cell.a3.norm();
    }
    else
    {
        ModuleBase::WARNING_QUIT("Dipole::ion_dipole", "direction is wrong!");
    }
    bmod = sqrt(pow(bvec[0],2) + pow(bvec[1],2) + pow(bvec[2],2));

    double ion_dipole = 0;
    double elec_dipole = 0;

    if(GlobalV::DIPOLE)
    {
        ion_dipole = cal_ion_dipole(cell, bmod);
        elec_dipole = cal_elec_dipole(cell, pwb, nspin, rho, bmod);
        tot_dipole = ion_dipole - elec_dipole;

        // energy correction
        etotefield = - ModuleBase::e2 * (eamp - 0.5 * tot_dipole) * tot_dipole * cell.omega / ModuleBase::FOUR_PI;
    }
    else
    {
        ion_dipole = cal_ion_dipole(cell, bmod);
        tot_dipole = ion_dipole - elec_dipole;

        // energy correction
        etotefield = - ModuleBase::e2 * eamp * tot_dipole * cell.omega / ModuleBase::FOUR_PI;
    }

    const double length = (1.0 - eopreg) * latvec * cell.lat0;
    const double vamp = ModuleBase::e2 * (eamp - tot_dipole) * length;

    GlobalV::ofs_running << "\n\n Adding external electric field: " << std::endl;
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Computed dipole along edir", edir);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Elec. dipole (Ry a.u.)", elec_dipole);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Ion dipole (Ry a.u.)", ion_dipole);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Total dipole (Ry a.u.)", tot_dipole);
    if( abs(eamp) > 0.0) 
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Amplitute of Efield (Hartree)", eamp);
    }
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Potential amplitute (Ry)", vamp);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "Total length (Bohr)", length);

    // dipole potential
    const int nspin0 = (nspin == 2) ? 2 : 1;

    for (int ir = 0; ir < pwb.nrxx; ++ir)
    {
        int i = ir / (pwb.ncy * pwb.nczp);
        int j = ir / pwb.nczp - i * pwb.ncy;
        int k = ir % pwb.nczp + pwb.nczp_start;
        double x = (double)i / pwb.ncx;
        double y = (double)j / pwb.ncy;
        double z = (double)k / pwb.ncz;
        ModuleBase::Vector3<double> pos(x, y, z);

        double saw = saw_function(emaxpos, eopreg, pos[edir]);

        for (int is = 0; is < nspin0; is++)
        {
            v(is, ir) = saw;
        }
    }

    double fac = ModuleBase::e2 * (eamp - tot_dipole) * cell.lat0 / bmod;

    ModuleBase::timer::tick("Dipole", "add_efield");
    return v * fac;
}


//=======================================================
// calculate dipole density in surface calculations
//=======================================================
double Dipole::cal_ion_dipole(const UnitCell &cell, const double &bmod)
{
    double ion_dipole = 0;
    for(int it=0; it<cell.ntype; ++it)
    {
        double sum = 0;
        for(int ia=0; ia<cell.atoms[it].na; ++ia)
        {
            sum += saw_function(emaxpos, eopreg, cell.atoms[it].taud[ia][edir]);
        }
        ion_dipole += sum * cell.atoms[it].zv;
    }
    ion_dipole *= cell.lat0 / bmod * ModuleBase::FOUR_PI / cell.omega;

    return ion_dipole;
}

double Dipole::cal_elec_dipole(const UnitCell &cell, 
                            PW_Basis &pwb, 
                            const int &nspin, 
                            const double *const *const rho,
                            const double &bmod)
{
    double elec_dipole = 0;
    const int nspin0 = (nspin == 2) ? 2 : 1;

    for (int ir = 0; ir < pwb.nrxx; ++ir)
    {
        int i = ir / (pwb.ncy * pwb.nczp);
        int j = ir / pwb.nczp - i * pwb.ncy;
        int k = ir % pwb.nczp + pwb.nczp_start;
        double x = (double)i / pwb.ncx;
        double y = (double)j / pwb.ncy;
        double z = (double)k / pwb.ncz;
        ModuleBase::Vector3<double> pos(x, y, z);

        double saw = saw_function(emaxpos, eopreg, pos[edir]);

        for (int is = 0; is < nspin0; is++)
        {
            elec_dipole += rho[is][ir] * saw;
        }
    }

    Parallel_Reduce::reduce_double_pool(elec_dipole);
    elec_dipole *= cell.lat0 / bmod * ModuleBase::FOUR_PI / pwb.ncxyz;

    return elec_dipole;
}

double Dipole::saw_function(const double &a, const double &b, const double &x)
{
    assert(x>=0);
    assert(x<=1);

    const double fac = 1 - b;

    if( x < a )
    {
        return x - a + 0.5 * fac;
    }
    else if( x > (a+b))
    {
        return x - a - 1 + 0.5 * fac;
    }
    else
    {
        return 0.5 * fac - fac * (x - a) / b;
    }
}

void Dipole::compute_force(const UnitCell &cell, ModuleBase::matrix &fdip)
{
    if(GlobalV::DIPOLE)
    {
        int iat = 0;
        for(int it=0; it<cell.ntype; ++it)
        {
            for(int ia=0; ia<cell.atoms[it].na; ++ia)
            {
                for(int jj=0; jj<3; ++jj)
                {
                    fdip(iat, jj) = ModuleBase::e2 * (eamp - tot_dipole) * cell.atoms[it].zv * bvec[jj] / bmod;
                }
                ++iat;
            }
        }
    }
    else
    {
        int iat = 0;
        for(int it=0; it<cell.ntype; ++it)
        {
            for(int ia=0; ia<cell.atoms[it].na; ++ia)
            {
                for(int jj=0; jj<3; ++jj)
                {
                    fdip(iat, jj) = ModuleBase::e2 * eamp * cell.atoms[it].zv * bvec[jj] / bmod;
                }
                ++iat;
            }
        }
    }
}