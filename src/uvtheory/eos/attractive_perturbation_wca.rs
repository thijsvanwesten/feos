use super::hard_sphere_wca::{diameter_wca, dimensionless_diameter_q_wca};
use crate::uvtheory::eos::CombinationRule;
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use ndarray::Array1;
use ndarray::Array2;
use num_dual::DualNum;
use std::{f64::consts::PI, fmt, sync::Arc};

const C_WCA: [[f64; 6]; 6] = [
    [
        -0.2622378162,
        0.6585817423,
        5.5318022309,
        0.6902354794,
        -3.6825190645,
        -1.7263213318,
    ],
    [
        -0.1899241690,
        -0.5555205158,
        9.1361398949,
        0.7966155658,
        -6.1413017045,
        4.9553415149,
    ],
    [
        0.1169786415,
        -0.2216804790,
        -2.0470861617,
        -0.3742261343,
        0.9568416381,
        10.1401796764,
    ],
    [
        0.5852642702,
        2.0795520346,
        19.0711829725,
        -2.3403594600,
        2.5833371420,
        432.3858674425,
    ],
    [
        -0.6084232211,
        -7.2376034572,
        19.0412933614,
        3.2388986513,
        75.4442555789,
        -588.3837110653,
    ],
    [
        0.0512327656,
        6.6667943569,
        47.1109947616,
        -0.5011125797,
        -34.8918383146,
        189.5498636006,
    ],
];

/// Constants for WCA u-fraction.
const CU_WCA: [f64; 3] = [1.4419, 1.1169, 16.8810];

/// Constants for WCA effective inverse reduced temperature.
const C2: [[f64; 2]; 3] = [
    [1.45805207053190E-03, 3.57786067657446E-02],
    [1.25869266841313E-04, 1.79889086453277E-03],
    [0.0, 0.0],
];

#[derive(Clone)]
pub struct AttractivePerturbationWCA {
    pub parameters: Arc<UVParameters>,
    pub combination_rule: CombinationRule,
}

impl fmt::Display for AttractivePerturbationWCA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for AttractivePerturbationWCA {
    /// Helmholtz energy for attractive perturbation, eq. 52
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        
        // Exact b21u? or based on vdws one-fluid temperature? Tests are based on vdws one-fluid...
        let exact_b21u = false;

        // Parameters and state
        let p = &self.parameters;
        let x = &state.molefracs;
        let t = state.temperature;
        let density = state.partial_density.sum();
        let n = p.ncomponents;

        // One-fluid parameters
        let (_rep_x, _att_x, sigma_x, _sigma3_vdw1f, epsilon_vdw1f, d_x, m_mix, 
            prefactor_b2, 
            prefactor_a1u) =
            one_fluid_properties(p, x, t);     

        let t_x = state.temperature / epsilon_vdw1f;    // VdW-1f temperature
        let rho_st = density * m_mix * sigma_x.powi(3); // dimensionless mixture density

        // Helmholtz energy
        let mut a = D::zero();         

        for i in 0..n {
            for j in 0..n {
                let t_ij = t / p.eps_k_ij[[i, j]];
                let rep_ij = p.rep_ij[[i,j]];
                let att_ij = p.att_ij[[i,j]];
        
                let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij));
                let q_ij_tx = dimensionless_diameter_q_wca(t_x, D::from(rep_ij), D::from(att_ij));
                
                // Perturbation term without its second-virial contribution...
                a += prefactor_a1u[[i,j]] / t * density * 
                correlation_integral_wca_noldl(rho_st, D::from(rep_ij), D::from(att_ij), d_x);

                // ...and its second-virial contribution
                let b21u_ij = prefactor_a1u[[i,j]] / t * ( if exact_b21u {
                    correlation_integral_wca_ldl(D::from(rep_ij), D::from(att_ij), q_ij)                    
                } else {
                    correlation_integral_wca_ldl(D::from(rep_ij), D::from(att_ij), q_ij_tx)
                } );   

                a += b21u_ij * density;

                // Interpolation term
                let b2pert_ij = prefactor_b2[[i,j]] * delta_b2(t_ij, rep_ij, att_ij, q_ij);
                let psi = D::one() - u_fraction_wca(D::from(rep_ij), rho_st);

                a += psi * (b2pert_ij - b21u_ij) * density;
            }
        }        
        // Helmholtz energy contribution divided by kT
        state.moles.sum() * a
    }
}

/// Correlation integral for first-order WCA perturbation term Mie fluids
fn correlation_integral_wca_noldl<D: DualNum<f64> + Copy>(
    rho: D,    
    rep: D,
    att: D,
    d: D,
) -> D {    
    let c = coefficients_wca(rep, att, d);    
    mie_prefactor(rep, att) * (c[0] * rho + c[1] * rho.powi(2) + c[2] * rho.powi(3))
        / (c[3] * rho + c[4] * rho.powi(2) + c[5] * rho.powi(3) + 1.0)
}

/// Low-density limit of correlation integral for first-order WCA perturbation term Mie fluids
fn correlation_integral_wca_ldl<D: DualNum<f64> + Copy>(    
    rep: D,
    att: D,
    q: D,
) -> D {
    let rm = (rep / att).powd((rep - att).recip());
    let mean_field_constant = mean_field_constant(rep, att, rm);
    (q.powi(3) - rm.powi(3)) / 3.0 - mean_field_constant
}

/// U-fraction for WCA division
fn u_fraction_wca<D: DualNum<f64> + Copy>(rep_x: D, reduced_density: D) -> D {
    (reduced_density * CU_WCA[0]
        + reduced_density.powi(2) * (rep_x.recip() * CU_WCA[2] + CU_WCA[1]))
        .tanh()
}

pub fn one_fluid_properties<D: DualNum<f64> + Copy>(
    p: &UVParameters,
    x: &Array1<D>,
    t: D,
) -> (D, D, D, D, D, D, D, Array2<D>, Array2<D>) {

    let d = diameter_wca(p, t);   

    let mut epsilon_k_vdw1f = D::zero();
    let mut sigma_vdw1f_3 = D::zero();
    let mut rep_x = D::zero();
    let mut att_x = D::zero();
    let mut d_x_st = D::zero();
    let mut m_mix = D::zero();
    let mut sigma_x = D::zero();
    let mut prefactor_b2 = Array2::<D>::zeros((p.ncomponents, p.ncomponents));
    let mut prefactor_a1u = Array2::<D>::zeros((p.ncomponents, p.ncomponents));

    for i in 0..p.ncomponents {

        let xi_mi = x[i]*p.m[i];

        // mixing rules preserving packing fracion and density of mixture
        m_mix += xi_mi;
        d_x_st += xi_mi * d[i].powi(3);
        sigma_x += xi_mi * p.sigma[i].powi(3);

        for j in 0..p.ncomponents {
            
            // Van-der-Waals-one-fluid mixing rules
            let pref = xi_mi * x[j] * p.m[j] * p.sigma_ij[[i, j]].powi(3);
            prefactor_b2[[i,j]] = pref;
            prefactor_a1u[[i,j]] = pref * p.eps_k_ij[[i, j]];
            sigma_vdw1f_3 += pref;
            epsilon_k_vdw1f += prefactor_a1u[[i, j]];

            // ... mixing rule for Mie exponents
            rep_x += x[i] * x[j] * p.rep_ij[[i, j]];
            att_x += x[i] * x[j] * p.att_ij[[i, j]];
        }
    }

    prefactor_a1u = prefactor_a1u * 2.0 * PI;
    epsilon_k_vdw1f = epsilon_k_vdw1f / sigma_vdw1f_3;
    sigma_vdw1f_3 = sigma_vdw1f_3 / m_mix.powi(2);
    sigma_x = (sigma_x/m_mix).powf(1.0/3.0);
    d_x_st = (d_x_st/m_mix).powf(1.0/3.0)  / sigma_x; // dimensionless

    (
        rep_x,
        att_x,
        sigma_x,
        sigma_vdw1f_3,
        epsilon_k_vdw1f,
        d_x_st,
        m_mix,
        prefactor_b2,
        prefactor_a1u
    )

        // let mut m_mix = D::zero();
        // let mut sigma_x = D::zero();
        // let mut d_x = D::zero();
        // let mut prefactor_f = Array2::<D>::zeros((n, n));
        // let mut prefactor_u = Array2::<D>::zeros((n, n));
        // let mut epsilon_vdw1f = D::zero();
        // let mut sigma3_vdw1f = D::zero();

        // for i in 0..n {
        //     let xi_mi = x[i] * p.m[i];

        //     m_mix += xi_mi;
        //     sigma_x += xi_mi * p.sigma[i].powi(3);
        //     d_x += xi_mi * d[i].powi(3);

        //     for j in 0..n {
        //         prefactor_f[[i,j]] = xi_mi * x[j] * p.m[j] * p.sigma_ij[[i, j]].powi(3);
        //         let pref = prefactor_f[[i,j]] * p.eps_k_ij[[i, j]];
        //         prefactor_u[[i, j]] = pref * 2.0 * PI / t;
        //         epsilon_vdw1f += pref;
        //         sigma3_vdw1f += prefactor_f[[i,j]];
        //     }
        // }

        // epsilon_vdw1f /= sigma3_vdw1f;
        // sigma_x = (sigma_x / m_mix).powf(1.0 / 3.0);
        // d_x = (d_x / m_mix).powf(1.0 / 3.0) / sigma_x;   

}

// Coefficients for IWCA from eq. (S55)
fn coefficients_wca<D: DualNum<f64> + Copy>(rep: D, att: D, d: D) -> [D; 6] {
    let rep_inv = rep.recip();
    let rs_x = (rep / att).powd((rep - att).recip());
    let tau_x = -d + rs_x;
    let c1 = rep_inv.powi(2) * C_WCA[0][2]
        + C_WCA[0][0]
        + rep_inv * C_WCA[0][1]
        + (rep_inv.powi(2) * C_WCA[0][5] + rep_inv * C_WCA[0][4] + C_WCA[0][3]) * tau_x;
    let c2 = rep_inv.powi(2) * C_WCA[1][2]
        + C_WCA[1][0]
        + rep_inv * C_WCA[1][1]
        + (rep_inv.powi(2) * C_WCA[1][5] + rep_inv * C_WCA[1][4] + C_WCA[1][3]) * tau_x;
    let c3 = rep_inv.powi(2) * C_WCA[2][2]
        + C_WCA[2][0]
        + rep_inv * C_WCA[2][1]
        + (rep_inv.powi(2) * C_WCA[2][5] + rep_inv * C_WCA[2][4] + C_WCA[2][3]) * tau_x;
    let c4 = rep_inv.powi(2) * C_WCA[3][2]
        + C_WCA[3][0]
        + rep_inv * C_WCA[3][1]
        + (rep_inv.powi(2) * C_WCA[3][5] + rep_inv * C_WCA[3][4] + C_WCA[3][3]) * tau_x;
    let c5 = rep_inv.powi(2) * C_WCA[4][2]
        + C_WCA[4][0]
        + rep_inv * C_WCA[4][1]
        + (rep_inv.powi(2) * C_WCA[4][5] + rep_inv * C_WCA[4][4] + C_WCA[4][3]) * tau_x;
    let c6 = rep_inv.powi(2) * C_WCA[5][2]
        + C_WCA[5][0]
        + rep_inv * C_WCA[5][1]
        + (rep_inv.powi(2) * C_WCA[5][5] + rep_inv * C_WCA[5][4] + C_WCA[5][3]) * tau_x;

    [c1, c2, c3, c4, c5, c6]
}

fn delta_b2<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64, q: D) -> D {
    // calculates dB2 / m^2 / sigma^3 
    let rm = (rep / att).powf(1.0 / (rep - att)); // Check mixing rule!!
    let rc = 5.0;
    let alpha = mean_field_constant(rep, att, rc);
    let beta = reduced_temperature.recip();
    let y = beta.exp() - 1.0;
    let yeff = y_eff(reduced_temperature, rep, att);
    -(yeff * (rc.powi(3) - rm.powi(3)) / 3.0 + y * (-q.powi(3) + rm.powi(3)) / 3.0 + beta * alpha)
        * 2.0
        * PI
}

fn y_eff<D: DualNum<f64> + Copy>(reduced_temperature: D, rep: f64, att: f64) -> D {
    // optimize: move this part to parameter initialization
    let rc = 5.0;
    let rs = (rep / att).powf(1.0 / (rep - att));
    let c0 = 1.0
        - 3.0 * (mean_field_constant(rep, att, rs) - mean_field_constant(rep, att, rc))
            / (rc.powi(3) - rs.powi(3));
    let c1 = C2[0][0] + C2[0][1] / rep;
    let c2 = C2[1][0] + C2[1][1] / rep;
    let c3 = C2[2][0] + C2[2][1] / rep;

    //exponents
    let a = 1.05968091375869;
    let b = 3.41106168592999;
    let c = 0.0;
    // (S58)
    let beta = reduced_temperature.recip();
    let beta_eff = beta
        * (-(beta.powf(a) * c1 + beta.powf(b) * c2 + beta.powf(c) * c3 + 1.0).recip() * c0 + 1.0);
    beta_eff.exp() - 1.0
}


fn residual_virial_coefficient<D: DualNum<f64> + Copy>(p: &UVParameters, x: &Array1<D>, t: D) -> D {
    let mut delta_b2bar = D::zero();

    for i in 0..p.ncomponents {
        let xi = x[i];

        for j in 0..p.ncomponents {
            //let q_ij = (q[i] / p.sigma[i] + q[j] / p.sigma[j]) * 0.5;
            let t_ij = t / p.eps_k_ij[[i, j]];
            let rep_ij = p.rep_ij[[i, j]];
            let att_ij = p.att_ij[[i, j]];

            let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij));

            // Recheck mixing rule!
            delta_b2bar +=
                xi * x[j] * p.sigma_ij[[i, j]].powi(3) * delta_b2(t_ij, rep_ij, att_ij, q_ij);
        }
    }
    delta_b2bar
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::eos::CombinationRule;
    use crate::uvtheory::parameters::utils::{methane_parameters, test_parameters_mixture};
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_attractive_perturbation() {
        // m = 24, t = 4.0, rho = 1.0
        let moles = arr1(&[2.0]);
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = methane_parameters(24.0, 6.0);
        let pt = AttractivePerturbationWCA {
            parameters: Arc::new(p.clone()),
            combination_rule: CombinationRule::OneFluidPsi,
        };
        let state = StateHD::new(
            reduced_temperature * p.epsilon_k[0],
            reduced_volume * p.sigma[0].powi(3),
            moles.clone(),
        );
        let x = &state.molefracs;

        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x, 
            _m_mix, _prefactor_f, _prefactor_u) =
            one_fluid_properties(&p, &state.molefracs, state.temperature);
        dbg!(epsilon_k_x);
        let t_x = state.temperature / epsilon_k_x;
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);
        
        dbg!(t_x);
        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        // let b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x)
        //     / p.sigma[0].powi(3);
        // assert_relative_eq!(b21u.re(), -1.02233215790525, epsilon = 1e-12);

        let i_wca_noldl = correlation_integral_wca_noldl(rho_x, rep_x, att_x, d_x);
        let i_wca_ldl = correlation_integral_wca_ldl(rep_x, att_x, q_vdw);
        let i_wca = i_wca_noldl + i_wca_ldl;

        let delta_a1u = state.partial_density.sum() / t_x * i_wca * 2.0 * PI * weighted_sigma3_ij;

        assert_relative_eq!(delta_a1u.re(), -1.52406840346272, epsilon = 1e-6);

        let u_fraction_wca = u_fraction_wca(
            rep_x,
            state.partial_density.sum() * (x * &p.sigma.mapv(|s| s.powi(3))).sum(),
        );

        let b2bar = residual_virial_coefficient(&p, x, state.temperature) / p.sigma[0].powi(3);
        dbg!(b2bar);
        assert_relative_eq!(b2bar.re(), -1.09102560732964, epsilon = 1e-12);
        dbg!(u_fraction_wca);

        assert_relative_eq!(u_fraction_wca.re(), 0.997069754340431, epsilon = 1e-5);

        // let a_test = delta_a1u
        //     + (-u_fraction_wca + 1.0)
        //         * (b2bar - b21u)
        //         * p.sigma[0].powi(3)
        //         * state.partial_density.sum();
        // dbg!(a_test);
        dbg!(state.moles.sum());
        let a = pt.helmholtz_energy(&state) / moles[0];
        dbg!(a.re());

        assert_relative_eq!(-1.5242697155023, a.re(), epsilon = 1e-5);
    }

    #[test]
    fn test_attractive_perturbation_wca_mixture() {
        let moles = arr1(&[0.40000000000000002, 0.59999999999999998]);
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;
        let reduced_volume = (moles[0] + moles[1]) / reduced_density;

        let p = test_parameters_mixture(
            arr1(&[1.0, 1.0]),
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 1.0]),
            arr1(&[1.0, 0.5]),
        );
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x, 
            _m_mix, _prefactor_f, _prefactor_u) =
            one_fluid_properties(&p, &state.molefracs, state.temperature);

        // u-fraction
        let phi_u = u_fraction_wca(rep_x, reduced_density);
        assert_relative_eq!(phi_u, 0.99750066585468078, epsilon = 1e-6);

        // Delta B21u
        let t_x = state.temperature / epsilon_k_x;

        dbg!(t_x.re());

        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        dbg!(q_vdw.re());
        // let delta_b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x);
        // dbg!(delta_b21u);
        // assert_relative_eq!(delta_b21u, -3.9309384983526585, epsilon = 1e-6);

        // delta a1u
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);

        let i_wca_noldl = correlation_integral_wca_noldl(rho_x, rep_x, att_x, d_x);
        let i_wca_ldl = correlation_integral_wca_ldl(rep_x, att_x, q_vdw);
        let i_wca = i_wca_noldl + i_wca_ldl;            

        let delta_a1u = state.partial_density.sum() / state.temperature
            * i_wca
            * 2.0
            * PI
            * weighted_sigma3_ij
            * epsilon_k_x;

        assert_relative_eq!(delta_a1u, -4.7678301069070645, epsilon = 1e-6);

        // Second virial coefficient

        let delta_b2 = residual_virial_coefficient(&p, &state.molefracs, state.temperature)
            / p.sigma[0].powi(3);

        dbg!(delta_b2);
        assert_relative_eq!(delta_b2, -4.7846399638747954, epsilon = 1e-6);
        // Full attractive contribution
        let pt = AttractivePerturbationWCA {
            parameters: Arc::new(p),
            combination_rule: CombinationRule::OneFluidPsi,
        };

        let a = pt.helmholtz_energy(&state) / (moles[0] + moles[1]);

        assert_relative_eq!(a, -4.7697504236074844, epsilon = 1e-5);
    }

    #[test]

    fn test_attractive_perturbation_wca_mixture_different_sigma() {
        let moles = arr1(&[0.40000000000000002, 0.59999999999999998]);
        let reduced_temperature = 1.5;
        let density = 0.10000000000000001;
        let volume = 1.0 / density;
        let p = test_parameters_mixture(
            arr1(&[1.0, 1.0]),
            arr1(&[12.0, 12.0]),
            arr1(&[6.0, 6.0]),
            arr1(&[1.0, 2.0]),
            arr1(&[1.0, 0.5]),
        );

        let state = StateHD::new(reduced_temperature, volume, moles.clone());
        let (rep_x, att_x, sigma_x, weighted_sigma3_ij, epsilon_k_x, d_x, 
            _m_mix, _prefactor_f, _prefactor_u) =
            one_fluid_properties(&p, &state.molefracs, state.temperature);
        // u-fraction
        let density = state.partial_density.sum();
        let x = &state.molefracs;
        let phi_u = u_fraction_wca(rep_x, density * (x * &p.sigma.mapv(|s| s.powi(3))).sum());
        assert_relative_eq!(phi_u, 0.89210738762113795, epsilon = 1e-5);
        // delta b2

        let b2bar = residual_virial_coefficient(&p, x, state.temperature) / p.sigma[0].powi(3);
        assert_relative_eq!(b2bar.re(), -12.106977583257606, epsilon = 1e-12);

        //delta b21u
        let t_x = state.temperature / epsilon_k_x;
        let q_vdw = dimensionless_diameter_q_wca(t_x, rep_x, att_x);
        // let delta_b21u = delta_b12u(t_x, mean_field_constant_x, weighted_sigma3_ij, q_vdw, rm_x);
        // assert_relative_eq!(delta_b21u, -10.841841323394299, epsilon = 1e-6);

        // let a_ufrac = (-phi_u + 1.0) * (b2bar - delta_b21u) * density;
        // assert_relative_eq!(a_ufrac, -0.0136498856091876, epsilon = 1e-6);
        // // delta b20
        dbg!(d_x.re());

        // delta a1u
        let rho_x = state.partial_density.sum() * sigma_x.powi(3);
        assert_relative_eq!(d_x, 0.95196953178057431, epsilon = 1e-6);

        let i_wca_noldl = correlation_integral_wca_noldl(rho_x, rep_x, att_x, d_x);
        let i_wca_ldl = correlation_integral_wca_ldl(rep_x, att_x, q_vdw);
        let i_wca = i_wca_noldl + i_wca_ldl;

        dbg!(weighted_sigma3_ij.re());
        dbg!(epsilon_k_x);
        let delta_a1u = state.partial_density.sum() / state.temperature
            * i_wca
            * 2.0
            * PI
            * weighted_sigma3_ij
            * epsilon_k_x;

        assert_relative_eq!(delta_a1u, -1.3182160310774731, epsilon = 1e-6);

        // Full attractive contribution
        let pt = AttractivePerturbationWCA {
            parameters: Arc::new(p),
            combination_rule: CombinationRule::OneFluidPsi,
        };
        let a = pt.helmholtz_energy(&state) / (moles[0] + moles[1]);
        assert_relative_eq!(a, -1.3318659166866607, epsilon = 1e-5);
    }
}
