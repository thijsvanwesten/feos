use crate::uvtheory::ufwca_tpt::hard_sphere_wca::{diameter_wca, dimensionless_diameter_q_wca};
use crate::uvtheory::parameters::*;
use feos_core::StateHD;
use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
// use std::{alloc::handle_alloc_error, f64::consts::PI, fmt};
use std::{f64::consts::PI, fmt};

// Coefficients for IWCA from eq. (S55)
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

/// Constants for WCA-uf u-fraction.
const CU_WCA: [f64; 5] = [2.3971, 2.5450, -1.0949, 3.1414, -1.1994];

#[derive(Clone)]
pub struct AttractivePerturbationWCAuf;

impl fmt::Display for AttractivePerturbationWCAuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Attractive Perturbation")
    }
}

impl AttractivePerturbationWCAuf {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> D {

        // Exact b21u? or based on vdws one-fluid temperature? Tests are based on vdws one-fluid...
        let exact_b21u = false; //true;

        // Parameters and state
        let p = &parameters;
        let x = &state.molefracs;
        let t = state.temperature;
        let density = state.partial_density.sum();
        let n = p.sigma.len();

        // One-fluid parameters
        let (
            _rep_x,
            _att_x,
            sigma_x,
            _sigma3_vdw1f,
            epsilon_vdw1f,
            epsilon_k_mf1f,
            d_x,
            m_mix,
            prefactor_b2,
            prefactor_a1u,
        ) = one_fluid_properties(p, x, t);

        let t_x = state.temperature / epsilon_vdw1f; // VdW-1f temperature
        let rho_st = density * m_mix * sigma_x.powi(3); // dimensionless mixture density
        
        // let d = diameter_wca(p, t);

        // Helmholtz energy  
        let mut a = D::zero();
        for i in 0..n {
            for j in 0..n {
                let t_ij = t / p.eps_k_ij[(i, j)];
                let rep_ij = p.rep_ij[(i, j)];
                let att_ij = p.att_ij[(i, j)];

                let rm_ij = (rep_ij / att_ij).powf(1.0 / (rep_ij - att_ij)); // Check mixing rule!!

                let q_ij = dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij));
                let q_ij_tx = dimensionless_diameter_q_wca(t_x, D::from(rep_ij), D::from(att_ij));                

                // u-perturbation term
                // let d_ij = (d[i]+d[j])/(p.sigma[i] + p.sigma[j]);
                let a1u = prefactor_a1u[(i, j)] * density / t * ( 
                    correlation_integral_u_wca_noldl(rho_st, D::from(rep_ij), D::from(att_ij), d_x) //d_ij
                    + (
                        if exact_b21u {
                            correlation_integral_u_wca_ldl(D::from(rep_ij), D::from(att_ij), q_ij)
                        } else {
                            correlation_integral_u_wca_ldl(D::from(rep_ij), D::from(att_ij), q_ij_tx)
                        }
                    )
                );

                // f-perturbation term
                let a1f = prefactor_b2[(i,j)] * density * ( 
                    correlation_integral_f_wca_noldl(rho_st, t/epsilon_k_mf1f, D::from(rep_ij))
                    + correlation_integral_f_wca_ldl(t_ij, rep_ij, q_ij, rm_ij)
                );

                let phiu = u_fraction_wcauf(D::from(rep_ij), rho_st);

                a += a1f + phiu * (a1u - a1f);                
            }
        }
        a * density
    }
}


/// u-fraction
fn u_fraction_wcauf<D: DualNum<f64> + Copy>(rep_x: D, reduced_density: D) -> D {
    let alpha = mean_field_constant(rep_x, D::one()*6.0, D::one());
    (reduced_density * CU_WCA[0]
        + reduced_density.powi(2) * (alpha.recip() * CU_WCA[2] + CU_WCA[1])
        + reduced_density.powi(4) * (alpha.recip() * CU_WCA[4] + CU_WCA[3])    
    )
    .tanh()
}

/// Low-density limit and remainder of correlation integral for first-order WCA perturbation term Mie fluids
/// 
fn correlation_integral_u_wca_ldl<D: DualNum<f64> + Copy>(rep: D, att: D, q: D) -> D {
    let rm = (rep / att).powd((rep - att).recip());
    let mean_field_constant = mean_field_constant(rep, att, rm);
    (q.powi(3) - rm.powi(3)) / 3.0 - mean_field_constant
}
fn correlation_integral_u_wca_noldl<D: DualNum<f64> + Copy>(rho_st: D, rep: D, att: D, d: D) -> D {
    let c = coefficients_wca(rep, att, d);
    mie_prefactor(rep, att) * (c[0] * rho_st + c[1] * rho_st.powi(2) + c[2] * rho_st.powi(3))
        / (c[3] * rho_st + c[4] * rho_st.powi(2) + c[5] * rho_st.powi(3) + 1.0)
}

/// Low-density limit and remainder of correlation integral for first-order WCA mayer-f perturbation term Mie fluids (2026 fit)
/// 
fn correlation_integral_f_wca_ldl<D: DualNum<f64> + Copy>(t_st: D, rep: f64, q: D, rm: f64) -> D {
    // calculates -If_ldl = dB2 / 2*pi*m^2*sigma^3 using RSAP model (van westen 2021)
    
    let beta = t_st.recip();
    let rep_inv = rep.recip();
    let ymf = beta.exp() - 1.0;

    let c = [
        -0.063550989, 6.206829830, -37.45829549, 40.72849774,
        1.519053409, 13.14989643, 85.35058674, 374.1906360, 
        0.693456220, 9.459946180, -53.28984218, 315.8199084, 
        0.007492596, 0.546171170, 7.979562575, -119.6126395 
        ];

    let c1 = c[0] + c[1]*rep_inv + c[2]*rep_inv.powi(2) + c[3]*rep_inv.powi(3);
    let c2 = c[4] + c[5]*rep_inv + c[6]*rep_inv.powi(2) + c[7]*rep_inv.powi(3);
    let c3 = c[8] + c[9]*rep_inv + c[10]*rep_inv.powi(2) + c[11]*rep_inv.powi(3);
    let c4 = c[12] + c[13]*rep_inv + c[14]*rep_inv.powi(2) + c[15]*rep_inv.powi(3);

    let mut sum = D::zero();
    let mut factorial = 1.0;
    
    for i in 1..16 {
        factorial = factorial * (i as f64);
        sum = sum + beta.powi(i) / factorial / (i as f64);
    }

    let twopi = 2.0*PI;
    
    ( ((q.powi(3) - rm.powi(3)) * twopi / 3.0 - c1)*ymf - sum*c2 - beta*c3 - beta.powi(2)*c4 ) / twopi
}

fn correlation_integral_f_wca_noldl<D: DualNum<f64> + Copy>(rho_st: D, t_st:D, rep: D) -> D {
    // calculates -(If-If_ldl)

    //------------------------
    // T*=[0.5-20.0], full rho* range, 8 < Mie_r < 30, APRD = 0.17% (uf-theory paper data-set)
    //------------------------          
    let cn = [
        -1.60899365e-01, -4.67405778e+00,  2.75626804e+01, -4.07374087e-01, 
        5.35396950e+00, -6.53308475e+01,  3.49298959e-01, -8.06868247e+00, 
        5.81368721e+01, -1.16724472e-01,  2.03465880e+00, -1.46408811e+01, 
        -5.68609210e-01,  1.63303206e+01, -8.83248953e+01,  1.17510646e+00, 
        -5.05811528e+01,  3.42902280e+02, -1.17263253e+00,  3.98160071e+01,
        -2.67687972e+02,  3.58966023e-01, -1.17082168e+01,  7.05916287e+01,
        3.78333583e-01, -8.90124822e+00,  4.99521692e+01, -1.09420629e+00,
        4.08706302e+01, -2.45491204e+02,  8.73506466e-01, -2.93664231e+01,
        1.88833258e+02, -2.90527029e-01,  8.27570253e+00, -4.60445699e+01 
        ];

    let mut l=0;
    let mut i1f_remainder = D::zero();

    let rho_st2 = rho_st*rho_st;
    let rho_pows = [rho_st, rho_st2, rho_st2*rho_st];
    let beta = t_st.recip();
    let beta2 = beta*beta;
    let tinv_pows = [beta, beta2, beta*beta2, beta2*beta2];
    let rep_inv = rep.recip();
    let repinv_pows = [D::one(), rep_inv, rep_inv*rep_inv];

    for k in 0..3 {
        for m in 0..4 {
            for n in 0..3 {                
                i1f_remainder +=  rho_pows[k] * tinv_pows[m] * repinv_pows[n] * cn[l];
                l+=1;
            }
        }
    }    
    i1f_remainder
}

/// One-fluid parameters
/// 
pub fn one_fluid_properties<D: DualNum<f64> + Copy>(
    p: &UVTheoryPars,
    x: &DVector<D>,
    t: D,
) -> (D, D, D, D, D, D, D, D, DMatrix<D>, DMatrix<D>) {
    let d = diameter_wca(p, t);

    let n = p.sigma.len();
    let mut epsilon_k_vdw1f = D::zero();
    let mut epsilon_k_mf1f = D::zero();
    let mut sigma_vdw1f_3 = D::zero();
    let mut rep_x = D::zero();
    let mut att_x = D::zero();
    let mut d_x_st = D::zero();
    let mut m_mix = D::zero();
    let mut sigma_x = D::zero();
    let mut prefactor_b2 = DMatrix::zeros(n, n);
    let mut prefactor_a1u = DMatrix::zeros(n, n);

    for i in 0..n {
        let xi_mi = x[i] * p.m[i];

        // mixing rules preserving packing fracion and density of mixture
        m_mix += xi_mi;
        d_x_st += xi_mi * d[i].powi(3);
        sigma_x += xi_mi * p.sigma[i].powi(3);

        for j in 0..n {
            // Van-der-Waals-one-fluid mixing rules
            let pref = xi_mi * x[j] * p.m[j] * p.sigma_ij[(i, j)].powi(3);
            prefactor_b2[(i, j)] = pref;
            prefactor_a1u[(i, j)] = pref * p.eps_k_ij[(i, j)];
            sigma_vdw1f_3 += pref;
            epsilon_k_vdw1f += prefactor_a1u[(i, j)];

            // Mixing rule for mayer-f perturbation term (B2 SW fluid)
            epsilon_k_mf1f += pref * ( t.recip() * p.eps_k_ij[(i, j)]).exp();

            // ... mixing rule for Mie exponents
            rep_x += x[i] * x[j] * p.rep_ij[(i, j)];
            att_x += x[i] * x[j] * p.att_ij[(i, j)];
        }
    }

    prefactor_a1u = prefactor_a1u * D::from(2.0 * PI);
    prefactor_b2 = prefactor_b2 * D::from(2.0 * PI);
    epsilon_k_vdw1f = epsilon_k_vdw1f / sigma_vdw1f_3;    
    epsilon_k_mf1f = t * ( epsilon_k_mf1f / sigma_vdw1f_3 ).ln();
    sigma_vdw1f_3 = sigma_vdw1f_3 / m_mix.powi(2);
    sigma_x = (sigma_x / m_mix).powf(1.0 / 3.0);
    d_x_st = (d_x_st / m_mix).powf(1.0 / 3.0) / sigma_x; // dimensionless

    (
        rep_x,
        att_x,
        sigma_x,
        sigma_vdw1f_3,
        epsilon_k_vdw1f,
        epsilon_k_mf1f,
        d_x_st,
        m_mix,
        prefactor_b2,
        prefactor_a1u,
    )
}

// Coefficients for Iu_wca from eq. (S55)
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


#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::{eos::AssociationModel, parameters::utils::{test_parameters, test_parameters_mixture}};
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_attractive_perturbation_wcauf_pure() {
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters(
            1.0,
            12.0,
            6.0,
            1.0,
            1.0,
            crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
        );
        let state = StateHD::new(reduced_temperature, reduced_volume, &dvector![1.0]);
        let a = AttractivePerturbationWCAuf.helmholtz_energy_density(&p, &state) / reduced_density;

        // assert_relative_eq!(a, 0.99996675884868391, epsilon = 1e-9); // phiu
        // assert_relative_eq!(a, -2.0985094801938375, epsilon = 1e-9);  // a1f
        assert_relative_eq!(a, -1.9305626014102031, epsilon = 1e-9);  

        
    }

    #[test]
    fn test_attractive_perturbation_wcauf_mixture_different_sigma_epsilon() {
        let molefracs = dvector![0.4, 0.6]; //dvector![1.0];

        let reduced_temperature = 1.5; //4.0; //1.5;
        let reduced_density = 0.1; //1.0; //0.1;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters_mixture(
            dvector![1.0, 1.0],
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 2.0],
            dvector![1.0, 0.5],
        );
        let p = UVTheoryPars::new(
            &p,
            crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
            AssociationModel::TVW,
        );

        // let p = test_parameters(
        //     1.0,
        //     24.0,
        //     6.0,
        //     1.0,
        //     1.0,
        //     crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
        // );

        let state = StateHD::new(reduced_temperature, reduced_volume, &molefracs);
        let a = AttractivePerturbationWCAuf.helmholtz_energy_density(&p, &state) / reduced_density;        

        // Full attractive contribution
        assert_relative_eq!(a, -1.3293844799948689, epsilon = 1e-5);
    }

    #[test]
    fn test_attractive_perturbation_wcauf_mixture_different_sigma_epsilon_m() {
        let molefracs = dvector![0.4, 0.6]; //dvector![1.0];

        let reduced_temperature = 1.5; //4.0; //1.5;
        let reduced_density = 0.1; //1.0; //0.1;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters_mixture(
            dvector![1.0, 1.2],
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 2.0],
            dvector![1.0, 0.5],
        );
        let p = UVTheoryPars::new(
            &p,
            crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
            AssociationModel::TVW,
        );

        // let p = test_parameters(
        //     1.0,
        //     24.0,
        //     6.0,
        //     1.0,
        //     1.0,
        //     crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT,
        // );

        let state = StateHD::new(reduced_temperature, reduced_volume, &molefracs);
        let a = AttractivePerturbationWCAuf.helmholtz_energy_density(&p, &state) / reduced_density;  

        // Full attractive contribution
        // assert_relative_eq!(a, -1.7824366683042956, epsilon = 1e-9);
        assert_relative_eq!(a, -1.7898868341910907, epsilon = 1e-9);
        
    }
}
