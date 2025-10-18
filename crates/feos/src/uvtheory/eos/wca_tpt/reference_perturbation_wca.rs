use super::hard_sphere_wca::{diameter_wca, dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a,packing_fraction_b};
use crate::uvtheory::parameters::*;
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
        let eta_a = packing_fraction_a(p, eta, state.temperature);
        let eta_b = packing_fraction_b(p, eta, state.temperature);
        let density = state.partial_density.sum();
        
        let mut a = D::zero();

        for i in 0..n {
            for j in 0..n {

                let rep_ij = p.rep_ij[(i,j)];
                let att_ij = p.att_ij[(i,j)];
                let rs_ij_3 = ( (rep_ij/att_ij).powf(1.0/(rep_ij - att_ij)) * p.sigma_ij[(i,j)] ).powi(3);

                // Additive hard-sphere fluid mixture as reference fluid for Mayer-f expansion
                let d_ij_3 = ((d[i] + d[j]) * 0.5).powi(3);
                
                // Exact low-density limit of correlation integral
                let t_ij = state.temperature / p.eps_k_ij[(i,j)];
                let q_ij_3 = ( dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij)) * p.sigma_ij[(i,j)] ).powi(3);
                let i0_ldl = d_ij_3 - q_ij_3;

                // MDA for correlation integral without low-density limit                
                let i0_noldl = ((-eta_a[(i,j)]*0.5 + 1.0) / (-eta_a[(i,j)] + 1.0).powi(3) - 1.0) * (-q_ij_3 + rs_ij_3)
                                - ((-eta_b[(i,j)]*0.5 + 1.0) / (-eta_b[(i,j)] + 1.0).powi(3) - 1.0) * (-d_ij_3 + rs_ij_3) ;
                                
                // Reduced Helmholtz energy per molecule / (2/3 PI rho)
                a -= x[i] * x[j] * p.m[i] * p.m[j] * ( i0_noldl + i0_ldl );
                    
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