use super::hard_sphere_wca::{diameter_wca, dimensionless_diameter_q_wca, packing_fraction};
use crate::uvtheory::parameters::*;
use crate::uvtheory::wca_tpt::hard_sphere_wca::{packing_fraction_a_ij, packing_fraction_b_ij};
use feos_core::StateHD;
// use nalgebra::{DMatrix, DVector};
use num_dual::DualNum;
use std::fmt;
use std::{f64::consts::PI};

#[derive(Clone)]
pub struct ReferencePerturbationWCA;

impl fmt::Display for ReferencePerturbationWCA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reference Perturbation")
    }
} 
    
impl ReferencePerturbationWCA {
    pub fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> D {
        let p = &parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let d = diameter_wca(p, state.temperature);        
        let eta = packing_fraction(&p.m, &state.partial_density, &d);
        let density = state.partial_density.sum();
        
        let mut a = D::zero();

        for i in 0..n {
            for j in 0..n {

                let rep = p.rep_ij[(i,j)];
                let att = p.att_ij[(i,j)];
                let rs_st = (rep/att).powf(1.0/(rep - att));
                let rs_st_3 = rs_st.powi(3);

                // Additive hard-sphere fluid mixture as reference fluid for Mayer-f expansion
                let d_st = ((d[i] + d[j]) * 0.5) /  p.sigma_ij[(i,j)];
                let d_st_3 = d_st.powi(3);
                
                // Exact low-density limit of correlation integral
                let t = state.temperature / p.eps_k_ij[(i,j)];
                let q_st_3 = ( dimensionless_diameter_q_wca(t, D::from(rep), D::from(att)) ).powi(3);
                let i0_ldl = d_st_3 - q_st_3;

                // MDA for correlation integral without low-density limit
                let (eta_a, _) = packing_fraction_a_ij(d_st, rs_st, rep, eta);
                let (eta_b, _) = packing_fraction_b_ij(d_st, rs_st, eta);
                let i0_noldl = ((-eta_a*0.5 + 1.0) / (-eta_a + 1.0).powi(3) - 1.0) * (-q_st_3 + rs_st_3)
                                - ((-eta_b*0.5 + 1.0) / (-eta_b + 1.0).powi(3) - 1.0) * (-d_st_3 + rs_st_3) ;
                                
                // Reduced Helmholtz energy per molecule / (2/3 PI rho)
                a -= x[i] * x[j] * p.m[i] * p.m[j] * p.sigma_ij[(i,j)].powi(3) * ( i0_noldl + i0_ldl );
                    
            }
        }
        // Reduced Helmholtz energy
        ( a * 2.0/3.0 * PI * density ) * density
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{test_parameters, test_parameters_mixture};
    use approx::assert_relative_eq;
    // use ndarray::arr1;
    use nalgebra::dvector;    

    #[test]
    fn test_delta_a0_wca_pure() {
        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters(
            1.0, 
            12.0, 
            6.0, 
            1.0, 
            1.0,
            crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT
        );       
        let state = StateHD::new(reduced_temperature, reduced_volume, &dvector![1.0]);
        let a = ReferencePerturbationWCA.helmholtz_energy_density(&p, &state) / reduced_density;

        assert_relative_eq!(a, 0.35951228642704941, epsilon = 1e-9);
    }
    #[test]
    fn test_delta_a0_wca_mixture() {

        let molefracs = dvector![0.4, 0.6];
        let reduced_temperature = 1.0;
        let reduced_density = 0.90000000000000002;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters_mixture(
            dvector![1.0,1.0],
            dvector![12.0,12.0],
            dvector![6.0,6.0],
            dvector![1.0,1.0],
            dvector![1.0,0.5],
        );
        let p = UVTheoryPars::new(&p, crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT);
        
        let state = StateHD::new(reduced_temperature, reduced_volume, &molefracs);
        let a = ReferencePerturbationWCA.helmholtz_energy_density(&p,&state) / reduced_density;

        assert_relative_eq!(a, 0.308268896386771, epsilon = 1e-6);
    }
    #[test]
    fn test_delta_a0_wca_mixture_different_sigma_chainlength() {

        let molefracs = dvector![0.4, 0.6];
        let reduced_temperature = 1.0;
        let reduced_density = 0.3;
        let reduced_volume = reduced_density.recip();

        let p = test_parameters_mixture(
            dvector![1.0,2.0],
            dvector![12.0,12.0],
            dvector![6.0,6.0],
            dvector![1.0,0.8],
            dvector![1.0,0.5],
        );
        let p = UVTheoryPars::new(&p, crate::uvtheory::Perturbation::WeeksChandlerAndersenTPT);
        
        let state = StateHD::new(reduced_temperature, reduced_volume, &molefracs);
        let a = ReferencePerturbationWCA.helmholtz_energy_density(&p,&state) / reduced_density;

        
        assert_relative_eq!(a, 8.1672292239339181E-002, epsilon = 1e-6);
    }
}