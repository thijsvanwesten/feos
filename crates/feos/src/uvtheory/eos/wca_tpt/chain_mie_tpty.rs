use super::hard_sphere_wca::{
    diameter_wca, dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a_ij,
    packing_fraction_b_ij, zeta,
};
use crate::uvtheory::eos::ChainContribution;
use crate::uvtheory::parameters::*;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::fmt;
use std::{f64::consts::FRAC_PI_6, f64::consts::PI};

const PAR: [f64; 19] = [
    0.225625,
    4.796965,
    32.221125,
    0.125222,
    -5.610177,
    -27.318559,
    -0.047445,
    53.729552,
    -436.591809,
    0.274554,
    -26.467849,
    491.023027,
    5.877262,
    -32.468883,
    1.84479,
    31.8795,
    30.32762,
    -446.19542,
    720.456266,
];

#[derive(Clone)]
pub struct ChainMie {
    pub chain_contribution: ChainContribution,
}

impl fmt::Display for ChainMie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TPT1y Mie Chain Contribution")
    }
}

impl ChainMie {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> D {
        let p = &parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let m = &p.m;
        let density = state.partial_density.sum();
        let d = diameter_wca(p, state.temperature);
        let eta = packing_fraction(&p.m, &state.partial_density, &d);
        let zms = (-eta + 1.0).recip();
        let zms2 = zms * zms;

        // let z2t = (x * m * d.mapv(|di| di.powi(2))).sum() * FRAC_PI_6;
        let z2t = (0..n).fold(D::zero(), |z, i| z + x[i] * m[i] * d[i].powi(2)) * FRAC_PI_6;
        let z2 = density * z2t;
        let rho_st = density * (0..n).fold(D::zero(), |z, i| z + x[i] * m[i] * p.sigma[i].powi(3));

        let mut a = D::zero();

        // TPT1-y? Else TPT1.
        let l_tpt1y = match self.chain_contribution {
            ChainContribution::TPT1y => true,
            ChainContribution::TPT1 => false,
        };

        for i in 0..n {
            //-----------------
            // TPT1
            //-----------------

            // CCF WCA fluid (MF1 theory)
            let y_wca_sigma = y_wca_aroundcontact_mix(
                1.0,
                &p,
                eta,
                &state.partial_density,
                state.temperature,
                &d,
                i,
                i,
            );

            // CCF Mie fluid (uf-theory)
            let t_st = state.temperature / p.epsilon_k[i];
            let nu_inv = (p.rep[i]).recip();
            let nu_inv2 = nu_inv * nu_inv;

            // move PAR to constant, see top of file
            let c1 = D::one() * (PAR[0] + PAR[1] * nu_inv + PAR[2] * nu_inv2)
                + t_st.recip() * (PAR[3] + PAR[4] * nu_inv + PAR[5] * nu_inv2);
            let c2 = D::one() * (PAR[6] + PAR[7] * nu_inv + PAR[8] * nu_inv2)
                + t_st.recip() * (PAR[9] + PAR[10] * nu_inv + PAR[11] * nu_inv2);
            let aa = (PAR[14] + PAR[15] * nu_inv).abs();
            let bb = PAR[16] * nu_inv + PAR[17] * nu_inv2 + PAR[18] * nu_inv.powi(3);
            let c3 = (D::one() + t_st).ln() * bb + aa;

            let prefactor = D::one() + (-c3 * rho_st).exp() * (PAR[12] + PAR[13] * nu_inv);
            let phiu = prefactor * (c1 * rho_st + c2 * rho_st * rho_st).tanh();

            let y_sigma = y_wca_sigma * (D::one() + phiu * ((-t_st.recip()).exp() - 1.0));

            // Helmholtz energy
            a -= x[i] * (m[i] - 1.0) * y_sigma.ln();

            //-----------------
            // TPT1-y (homo-segmented)
            //-----------------
            if p.m[i] > 2.0 && l_tpt1y {
                let fac1 = (m[i] - 1.0) / m[i];
                let fac3 = (m[i] - 2.0) / m[i];

                // next-nearest-neighbour contribution WCA segments
                let bondlength_sq = (d[i].recip() * p.sigma[i]).powi(2);
                let bfac =
                    ((bondlength_sq * 4.0 + 1.0) / (bondlength_sq * 2.0)).sqrt() * p.sigma[i];
                let bfac_st =
                    ((bondlength_sq.powi(2) * 4.0 + bondlength_sq) / (bondlength_sq * 5.0)).sqrt();

                let par2 = [0.35749, -1.2591, 0.84841, -1.3420, 3.2487, -2.5243];
                let a3 = par2[0] + par2[1] * fac3 + par2[2] * fac3.powi(2);
                let b3 = par2[3] + par2[4] * fac3 + par2[5] * fac3.powi(2);

                let ynn_wca = D::one()
                    + bfac_st.powi(2) * (zms - 1.0)
                    + bfac * z2 * a3 * zms2
                    + (bfac * z2).powi(2) * b3 * zms.sqrt();

                // next-nearest-neighbour contribution Mie segments
                let par3 = [
                    8.187740E-004,
                    1.247571,
                    8.507247,
                    0.15022302,
                    0.86897494,
                    -2.0306142,
                    -9.7821165E-005,
                    2.3896468E-002,
                    1.2597622,
                    9.7603646E-004,
                ];
                let cc1 = par3[0] + par3[1] * fac1 + (par3[2] + par3[3] * fac1) * nu_inv;
                let cc2 = par3[4] + par3[5] * fac1 + (par3[6] + par3[7] * fac1) * nu_inv;
                let cc3 = par3[8] + par3[9] * nu_inv;

                let phiu_rdf = (rho_st * cc1 + rho_st.powi(2) * cc2).tanh();
                let ynn_att = phiu_rdf * ((-t_st.recip() * cc3).exp() - 1.0) + 1.0;

                // Helmholtz energy contribution
                a -= x[i] * (m[i] - 2.0) * bfac_st.powf(1.5) * fac3 * ynn_wca.ln();
                a -= x[i]
                    * (m[i] - 2.0)
                    * bfac_st.powf(1.5)
                    * ((p.m[i] - 2.0) / (p.m[i] - 1.0))
                    * ynn_att.ln();
            }
        }
        density * a
    }
}

// CCF of two WCA monomers of index i and j in a WCA fluid mixture at reduced distance r/sigma.
// NB temperature is dimensional.
fn y_wca_aroundcontact_mix<D: DualNum<f64> + Copy>(
    r_st: f64,
    p: &UVTheoryPars,
    eta: D,
    partial_density: &DVector<D>,
    temperature: D,
    dhs: &DVector<D>,
    i: usize,
    j: usize,
) -> D {
    let mseg = &p.m;
    let rep = p.rep_ij[(i, j)];
    let att = p.att_ij[(i, j)];
    let sigma = p.sigma_ij[(i, j)];
    let t_st = temperature / p.eps_k_ij[(i, j)];
    let d_hs = (dhs[i] + dhs[j]) * 0.5;

    let yhs_r = y_hf(
        &partial_density,
        &mseg,
        &dhs,
        i,
        j,
        d_hs.recip() * r_st * sigma,
    );
    let yhs_d = y_hf(&partial_density, &mseg, &dhs, i, j, D::one());

    let d_st = d_hs / sigma;
    let d_st_3 = d_st.powi(3);
    let q_st_3 = dimensionless_diameter_q_wca(t_st, D::one() * rep, D::one() * att).powi(3);
    let rm_st = (rep / att).powf((rep - att).recip());
    let rm_st_3 = rm_st.powi(3);

    // Effective packing fractions (+derivatives) for first-order Mayer-f perturbation term in Helmholz energy da01
    let (eta_a, eta_a_eta) = packing_fraction_a_ij(d_st, rm_st, rep, eta);
    let (eta_b, eta_b_eta) = packing_fraction_b_ij(d_st, rm_st, eta);

    let zms_a = D::one() / (D::one() - eta_a);
    let zms_b = D::one() / (D::one() - eta_b);
    let zms_a3 = zms_a * zms_a * zms_a;
    let zms_b3 = zms_b * zms_b * zms_b;
    let yhs_a = (D::one() - eta_a / 2.0) * zms_a3;
    let yhs_b = (D::one() - eta_b / 2.0) * zms_b3;
    let yhs_a_eta =
        (-eta_a_eta / 2.0) * zms_a3 + (D::from(3.0) - eta_a * 1.5) * eta_a_eta * zms_a3 * zms_a;
    let yhs_b_eta =
        (-eta_b_eta / 2.0) * zms_b3 + (D::from(3.0) - eta_b * 1.5) * eta_b_eta * zms_b3 * zms_b;

    // Effective packing fraction specific to this routine
    let eta_c = packing_fraction_c_ij(d_st, rm_st, rep, eta);
    let yhs_c = (D::one() - eta_c / 2.0) / (D::one() - eta_c).powi(3);

    // y01
    let integral1 = -yhs_c * q_st_3 / d_st_3;
    let fac1 = (-q_st_3 + rm_st_3) / d_st_3;
    let fac2 = (-d_st_3 + rm_st_3) / d_st_3;
    let integral2 = -eta * (yhs_a_eta * fac1 - yhs_b_eta * fac2) - (yhs_a * fac1 - yhs_b * fac2);
    let yhs_cs = (D::one() - eta * 0.5) / (D::one() - eta).powi(3);
    let y01 = integral1 + integral2 + yhs_cs;

    // First-order Mayer-f perturbation expansion ln(y0(r)) about ln(y^HS_d(r))
    yhs_r * (y01 / yhs_d).exp()
}

// Effective packing fraction specific to CCF of WCA fluid around contact
// pub(super)
fn packing_fraction_c_ij<D: DualNum<f64> + Copy>(dhs_st: D, rmin_st: f64, rep: f64, eta: D) -> D {
    let tau = -dhs_st + rmin_st;
    let rep_inv = 1.0 / rep;
    let rep_inv2 = rep_inv * rep_inv;

    let para_ic = [
        -2.43121181e-01,
        -4.42246830e+00,
        1.27240515e+00,
        -1.76709844e+01,
        -2.59293256e-01,
        1.91474433e+01,
        8.67908337e-01,
        -2.13124347e+01,
        -2.70748992e-01,
        -7.57290826e+01,
        2.96315756e+00,
        -2.80211182e+01,
        -1.43679473e-01,
        9.74930050e+01,
        1.68719700e+00,
        3.80164876e+01,
        3.04912631e+01,
        -6.94411993e+01,
        3.02445041e+02,
        -3.42554082e+02,
        2.21010882e+01,
        8.59563268e+01,
        -6.35352038e+01,
        -5.05404224e+01,
    ];

    let c1 = tau * (para_ic[0] + para_ic[1] * rep_inv + para_ic[16] * rep_inv2)
        + tau.powi(2) * (para_ic[2] + para_ic[3] * rep_inv + para_ic[20] * rep_inv2);
    let c2 = tau * (para_ic[4] + para_ic[5] * rep_inv + para_ic[17] * rep_inv2)
        + tau.powi(2) * (para_ic[6] + para_ic[7] * rep_inv + para_ic[21] * rep_inv2);
    let c3 = tau * (para_ic[8] + para_ic[9] * rep_inv + para_ic[18] * rep_inv2)
        + tau.powi(2) * (para_ic[10] + para_ic[11] * rep_inv + para_ic[22] * rep_inv2);
    let c4 = tau * (para_ic[12] + para_ic[13] * rep_inv + para_ic[19] * rep_inv2)
        + tau.powi(2) * (para_ic[14] + para_ic[15] * rep_inv + para_ic[23] * rep_inv2);

    eta + eta * c1 + eta.powi(2) * c2 + eta.powi(3) * c3 + eta.powi(4) * c4
}

/// HF model for hard-sphere cavity function y(xx) of two hard spheres
/// of index l and m in a HS mixture. Model of Ben-Amotz, Herschbach and de Souza.
fn y_hf<D: DualNum<f64> + Copy>(
    partial_density: &DVector<D>,
    mseg: &DVector<f64>,
    d: &DVector<D>,
    l: usize,
    m: usize,
    xx: D,
) -> D {
    let n = mseg.len();

    let zeta = zeta(mseg, partial_density, d);
    let z0 = zeta[0];
    let z1 = zeta[1];
    let z2 = zeta[2];
    let z3 = zeta[3];
    let zms = -z3 + 1.0;
    let zms = zms.recip();
    let zms2 = zms * zms;
    let zms3 = zms2 * zms;

    //-------------------
    // CCF at HS contact
    //-------------------
    // let mut y1 = DMatrix::<D>::zeros(n, n);
    // for i in 0..n {
    //     for j in 0..n {
    //         let dij = d[i] * d[j] * (d[i] + d[j]).recip();
    //         y1[(i, j)] = zms + zms2 * 3.0 * z2 * dij + zms3 * 2.0 * (z2 * dij).powi(2);
    //     }
    // }

    let y1 = DMatrix::from_fn(n, n, |i, j| {
        let dij = d[i] * d[j] * (d[i] + d[j]).recip();
        zms + zms2 * 3.0 * z2 * dij + zms3 * 2.0 * (z2 * dij).powi(2)
    });

    //-------------------
    // derivative of log(CCF) to r at zero separation
    //-------------------
    let mut dlog_y0 = DVector::<D>::zeros(n);
    for i in 0..n {
        for k in 0..n {
            let dik = (d[i] + d[k]) * 0.5;
            dlog_y0[i] -= partial_density[k] * mseg[k] * y1[(i, k)] * dik * dik * PI;
        }
    }

    // let dlog_y0 = DVector::from_fn(n, |i, _| {
    //     (0..n).fold(D::zero(),|z,k| {
    //         let dik = (d[i] + d[k]) * 0.5;
    //         z - partial_density[k] * mseg[k] * y1[(i, k)] * dik * dik * PI
    //     })
    // });

    //-------------------
    // log(CCF) at zero separation (from chemical potential of HS in a HS-mixture)
    //-------------------
    let z0 = z0 / z3;
    let z1 = z1 / z3;
    let z2 = z2 / z3;

    // find smallest monomer of pair lm (if equal irrelevant)
    let mut a_lm = (d[l] - d[m]) * 0.5;
    let k = if a_lm.re() > 1.0e-15 { m } else { l };
    a_lm = a_lm.abs();

    // density-independent parameters
    let dk = d[k];
    let c1 = D::one() + z2 * dk * 3.0 + D::one() * 3.0 * z1 * dk.powi(2) + z0 * dk.powi(3);
    let c2 = D::one() * (-3.0) - z2 * dk * 6.0
        + (z2 * z2 * 9.0 - z1 * 6.0) * dk * dk
        + (z1 * z2 * 6.0 - z0 * 2.0) * dk.powi(3);
    let c3 = D::one() * 3.0
        + z2 * dk * 3.0
        + (z1 * 3.0 - z2 * z2 * 12.0) * dk * dk
        + (z0 - z1 * z2 * 6.0 + z2.powi(3) * 8.0) * dk.powi(3);
    let c4 =
        D::one() * (-1.0) + D::one() * 3.0 * (z2 * dk).powi(2) - D::one() * 2.0 * (z2 * dk).powi(3);

    // density-dependent parameters
    let i1 = (zms3 - 1.0) * (1.0 / 3.0);
    let i2 = z3.powi(2) * (D::one() * 3.0 - z3) * zms3 / 6.0;
    let i3 = (z3 * z3 * z3 * zms3) / 3.0;
    let i4 = (z3 * (D::one() * 6.0 - z3 * 15.0 + z3 * z3 * 11.0) * zms3 / 6.0) + (-z3 + 1.0).ln();

    let log_y0 = c1 * i1 + c2 * i2 + c3 * i3 + c4 * i4;

    //-------------------
    // CCF at dimensionless distance xx = r/dhs
    //-------------------
    let dhs = (d[l] + d[m]) * 0.5;
    let lny1 = y1[(l, m)].ln();
    let blm = (dlog_y0[l] + dlog_y0[m]) * 0.5;
    let clm = (lny1 - log_y0 - blm * (dhs - a_lm * 2.0 + a_lm * a_lm / dhs))
        / (dhs.powi(3) - a_lm.powi(3) * 4.0 + a_lm.powi(4) * 3.0 / dhs);
    let dlm = a_lm * a_lm * (blm + clm * 3.0 * a_lm * a_lm);
    let alm = lny1 - blm * dhs - clm * dhs.powi(3) - dlm / dhs;

    (alm + blm * (xx * dhs) + clm * (xx * dhs).powi(3) + dlm / ((xx * dhs).re().max(1.0e-16))).exp()
}

/*
/// Hard-sphere cavity functio from MSPT
///  The cavity-correlation function of two (heteronuclear) HS cavities
///  of index i and j, at reduced distance xx, in a hard-sphere fluid
///  mixture, according to the Modified-Scaled-Particle-Theory (MSPT)
///  of Boublik (1986).
fn g_mspt<D: DualNum<f64> + Copy>(
    density: D,
    x: &DVector<D>,
    m: &DVector<f64>,
    d: &DVector<D>,
    i: usize,
    xx: D,
) -> D {
    let mbar = (x * m).sum();
    let rho_s = density * mbar;
    let x_s = (x * m).mapv(|e| e / mbar);

    let nu = rho_s * FRAC_PI_6 * (&x_s * d.mapv(|di| di.powi(3))).sum();
    let s = rho_s * PI * (&x_s * d.mapv(|di| di.powi(2))).sum();
    let q = rho_s * (&x_s * d.mapv(|di| di.powi(2))).sum() / 4.0;
    let r = rho_s * (&x_s * d).sum() * 0.5;

    let v_i = d[i].powi(3) * FRAC_PI_6;
    let s_i = d[i].powi(2) * PI;
    let r_i = d[i] * 0.5;

    let muhs_i = -(-nu + 1.0).ln()
        + (r_i * s + s_i * r + v_i * rho_s) / (-nu + 1.0)
        + (r_i.powi(2) * s.powi(2) + s_i * q * s * 2.0 + v_i * r * s * 6.0)
            / ((-nu + 1.0).powi(2) * 6.0)
        + (v_i * q * s.powi(2) * (-nu / 3.0 + 2.0)) / ((-nu + 1.0).powi(3) * 9.0);

    // !-------------------
    // ! chemical potential of Hard-Dimer,
    // ! formed from HS i and j at bond-lengt xx=2.0*l/(di+dj),
    // ! infinitely diluted in HS mixture
    // !-------------------
    let v_ij =
        (-xx.powi(3) / 8.0 * 2.0.powi(3) + 2.0 + xx * 0.75 * 4.0) * d[i].powi(3) * (PI / 12.0);
    let s_ij = (xx + 1.0) * d[i].powi(2) * PI;
    let r_ij = (xx / 2.0 + 1.0) * d[i] * 0.5;

    // dbg!(v_ij);
    // dbg!(s_ij);
    // dbg!(r_ij);

    let muhd_ij = -(-nu + 1.0).ln()
        + (r_ij * s + s_ij * r + v_ij * rho_s) / (-nu + 1.0)
        + (r_ij.powi(2) * s.powi(2) + s_ij * q * s * 2.0 + v_ij * r * s * 6.0)
            / ((-nu + 1.0).powi(2) * 6.0)
        + (v_ij * q * s.powi(2) * (-nu / 3.0 + 2.0)) / ((-nu + 1.0).powi(3) * 9.0);

    // !-------------------
    // ! cavity function of HS i and HS j at distance L,
    // ! in HS mixture
    // !-------------------
    (muhs_i * 2.0 - muhd_ij).exp()
}
 */

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{test_parameters, test_parameters_mixture};
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    /*
    #[test]
    fn test_y_wca_aroundcontact_mixture() {
        let moles = dvector![0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.1;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVTheoryPars::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.25, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();

        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_wca(&p, state.temperature);
        let eta = packing_fraction(&p.m, &state.partial_density, &d);

        let y_wca_00 = y_wca_aroundcontact_mix(
            1.0,
            &p,
            eta,
            &state.partial_density,
            state.temperature,
            &d,
            0,
            0
        );
        let y_wca_01 = y_wca_aroundcontact_mix(
            1.0,
            &p,
            eta,
            &state.partial_density,
            state.temperature,
            &d,
            0,
            1
        );
        let y_wca_11 = y_wca_aroundcontact_mix(
            1.0,
            &p,
            eta,
            &state.partial_density,
            state.temperature,
            &d,
            1,
            1
        );

        assert_eq!((y_wca_00, y_wca_01, y_wca_11), (1.2333573033407719,1.2202426938494642, 1.2097785725071151));
    }

    #[test]
    fn test_y_wca_aroundcontact_mixture2() {
        let moles = dvector![0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.1;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVTheoryPars::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.25, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        2.0, 12.0, 6.0, 1.0, 1.5, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();

        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_wca(&p, state.temperature);
        let eta = packing_fraction(&p.m, &state.partial_density, &d);

        let y_wca_00 = y_wca_aroundcontact_mix(
            1.0,
            &p,
            eta,
            &state.partial_density,
            state.temperature,
            &d,
            0,
            0
        );
        let y_wca_01 = y_wca_aroundcontact_mix(
            1.0,
            &p,
            eta,
            &state.partial_density,
            state.temperature,
            &d,
            0,
            1
        );
        let y_wca_11 = y_wca_aroundcontact_mix(
            1.0,
            &p,
            eta,
            &state.partial_density,
            state.temperature,
            &d,
            1,
            1
        );

        assert_eq!((y_wca_00, y_wca_01, y_wca_11), (1.3165641196539490,1.3050790505787109,1.2938262092789194));
    }
     */

    /*
    #[test]
    fn test_y_hf() {

        let moles = dvector![2.0]);
        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles[0] / reduced_density;

        // playing around with dual number to get derivatives:

        // // First derivative to temperature
        // let moles = dvector![Dual64::from_re(2.0)]);
        // let reduced_temperature = Dual64::new(2.0,1.0);
        // let reduced_density = Dual64::from_re(0.6);
        // let reduced_volume = moles[0] / reduced_density;

        // // second derivative to temperature
        // // let moles = dvector![Dual2_64::from_re(2.0)]);
        // let reduced_temperature = Dual2_64::from_re(2.0).derivative();
        // // let reduced_density = Dual2_64::from_re(0.6);
        // // let reduced_volume = moles[0] / reduced_density;

        // // mixed double derivative
        // // let moles = dvector![HyperDual64::from_re(2.0)]);
        // let reduced_temperature = HyperDual64::from_re(2.0).derivative1();
        // // let reduced_density = HyperDual64::from_re(0.6).derivative2();
        // // let reduced_volume = moles[0] / reduced_density;

        // // third derivative
        // // let moles = dvector![Dual3_64::from_re(2.0)]);
        // let reduced_temperature = Dual3_64::from_re(2.0).derivative();
        // // let reduced_density = Dual3_64::from_re(0.6);
        // // let reduced_volume = moles[0] / reduced_density;


        let p = test_parameters(1.0, 12.0, 6.0, 1.0, 1.0);
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_wca(&p, state.temperature);

        dbg!(&d);
        assert!(1.0==2.0);

        let y_hf = y_hf(
            &state.partial_density,
            &p.m,
            &d,
            0,
            0,
            d[0].recip() * p.sigma[0],
        );
        assert_eq!((y_hf.re()*1e10).round()/1e10, 2.1615376807);
    }

    #[test]
    fn test_eta_ab_mixture() {
        let moles = dvector![0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVTheoryPars::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        2.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let dhs = diameter_wca(&p, state.temperature);
        // assert_eq!(dhs[0],diameter_wca_i(&p, state.temperature, 0));
        // assert_eq!(dhs[1],diameter_wca_i(&p, state.temperature, 1));

        let eta = packing_fraction(&p.m, &state.partial_density, &dhs);

        let eta_a = packing_fraction_a(&p, eta, state.temperature);
        let eta_b = packing_fraction_b(&p, eta, state.temperature);

        let eta_a_11 = packing_fraction_a_ij(&p, eta, state.temperature, 0,0);
        let eta_b_11 = packing_fraction_b_ij(&p, eta, state.temperature, 0, 0);
        let eta_a_12 = packing_fraction_a_ij(&p, eta, state.temperature, 0,1);
        let eta_b_12 = packing_fraction_b_ij(&p, eta, state.temperature, 0, 1);
        let eta_a_22 = packing_fraction_a_ij(&p, eta, state.temperature, 1,1);
        let eta_b_22 = packing_fraction_b_ij(&p, eta, state.temperature, 1, 1);

        assert_eq!((eta_a[[0,0]],eta_b[[0,0]]),(eta_a_11.0,eta_b_11.0));
        assert_eq!((eta_a[[0,1]],eta_b[[0,1]]),(eta_a_12.0,eta_b_12.0));
        assert_eq!((eta_a[[1,1]],eta_b[[1,1]]),(eta_a_22.0,eta_b_22.0));

    }

    #[test]
    fn test_y_hf_mixture() {
        let moles = dvector![0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.6;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVTheoryPars::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        2.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_wca(&p, state.temperature);
        dbg!(&d);
        let y_hf_0 = y_hf(
            &state.partial_density,
            &p.m,
            &d,
            0,
            0,
            p.sigma[0] / d[0],
        );
        let y_hf_1 = y_hf(
            &state.partial_density,
            &p.m,
            &d,
            1,
            1,
            p.sigma[1] / d[1],
        );
        assert_eq!(y_hf_0, y_hf_1);
    }

    #[test]
    fn test_y_hf_mixture_2() {
        let moles = dvector![0.6, 0.4]) * 2.0;

        let reduced_temperature = 2.0;
        let reduced_density = 0.1;
        let reduced_volume = moles.sum() / reduced_density;

        let p = UVTheoryPars::new_binary(
            vec![
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.25, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
                PureRecord::new(
                    Identifier::default(),
                    1.0,
                    UVRecord::new(
                        1.0, 12.0, 6.0, 1.0, 1.0, None, None, None, None, None, None, None,
                    ),
                ),
            ],
            None,
        )
        .unwrap();
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());

        let d = diameter_wca(&p, state.temperature);
        dbg!(&d);
        let y_hf_00 = y_hf(
            &state.partial_density,
            &p.m,
            &d,
            0,
            0,
            p.sigma[0] / d[0],
        );
        let y_hf_11 = y_hf(
            &state.partial_density,
            &p.m,
            &d,
            1,
            1,
            p.sigma[1] / d[1],
        );
        let y_hf_01 = y_hf(
            &state.partial_density,
            &p.m,
            &d,
            0,
            1,
            (p.sigma[0]+p.sigma[1]) / (d[0]+d[1]),
        );
        assert_eq!((y_hf_00, y_hf_01, y_hf_11), (1.207305310,1.194122519,1.183603472));
    }
     */

    // /*
    #[test]
    fn test_a_ljchain() {
        let reduced_temperature = 2.0;
        let reduced_density = 0.6;

        let p = test_parameters(
            2.0,
            12.0,
            6.0,
            1.0,
            1.0,
            crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
        );
        let chain = ChainMie {
            chain_contribution: ChainContribution::TPT1y,
        };
        let state = StateHD::new(reduced_temperature, reduced_density.recip(), &dvector![1.0]);

        let a_chain = chain.helmholtz_energy_density(&p, &state) / reduced_density;
        dbg!(&a_chain);
        assert_relative_eq!(a_chain, -1.2671979739364336, epsilon = 1e-9);
    }

    #[test]
    fn test_a_miechain() {
        let reduced_temperature = 2.0;
        let reduced_density = 0.1;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters(
            5.0,
            18.0,
            6.0,
            1.0,
            1.0,
            crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
        );
        let chain = ChainMie {
            chain_contribution: ChainContribution::TPT1y,
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, &dvector![1.0]);

        let a_chain = chain.helmholtz_energy_density(&p, &state) / reduced_density;
        dbg!(&a_chain);
        assert_relative_eq!(a_chain, -0.90267009812299037, epsilon = 1e-9);
    }

    #[test]
    fn test_a_chain_mixture() {
        let molefracs = dvector![0.6, 0.4];

        let reduced_temperature = 2.0;
        let reduced_density = 0.1;
        let reduced_volume = 1.0 / reduced_density;

        let p = test_parameters_mixture(
            dvector![3.0, 2.0],
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.5],
            dvector![1.0, 1.25],
        );
        let p = UVTheoryPars::new(&p, crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT);

        let chain = ChainMie {
            chain_contribution: ChainContribution::TPT1y,
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, &molefracs);

        let a = chain.helmholtz_energy_density(&p, &state) / reduced_density;
        dbg!(&a);
        assert_relative_eq!(a, -0.22988783790867773, epsilon = 1e-9);
    }
    //  */
}
