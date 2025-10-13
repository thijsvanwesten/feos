use super::hard_sphere_wca::{
    diameter_wca, dimensionless_diameter_q_wca, packing_fraction, packing_fraction_a,
    packing_fraction_b,
};
use crate::uvtheory::parameters::*;
use feos_core::{HelmholtzEnergyDual, StateHD};
use num_dual::DualNum;
use std::fmt;
use std::{f64::consts::PI, sync::Arc};

#[derive(Clone)]
pub struct ReferencePerturbationWCA {
    pub parameters: Arc<UVParameters>,
}

impl fmt::Display for ReferencePerturbationWCA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reference Perturbation")
    }
}

impl<D: DualNum<f64> + Copy> HelmholtzEnergyDual<D> for ReferencePerturbationWCA {
    /// Helmholtz energy for perturbation reference (Mayer-f), eq. 29
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let n = p.sigma.len();
        let x = &state.molefracs;
        let d = diameter_wca(p, state.temperature);        
        let eta = packing_fraction(&p.m, &state.partial_density, &d);
        let eta_a = packing_fraction_a(p, eta, state.temperature);
        let eta_b = packing_fraction_b(p, eta, state.temperature);
        
        let mut a = D::zero();

        for i in 0..n {
            for j in 0..n {

                let rep_ij = p.rep_ij[[i, j]];
                let att_ij = p.att_ij[[i, j]];
                let rs_ij_3 = ( (rep_ij/att_ij).powf(1.0/(rep_ij - att_ij)) * p.sigma_ij[[i,j]] ).powi(3);

                // Additive hard-sphere fluid mixture as reference fluid for Mayer-f expansion
                let d_ij_3 = ((d[i] + d[j]) * 0.5).powi(3);
                
                // Exact low-density limit of correlation integral
                let t_ij = state.temperature / p.eps_k_ij[[i, j]];
                let q_ij_3 = ( dimensionless_diameter_q_wca(t_ij, D::from(rep_ij), D::from(att_ij)) * p.sigma_ij[[i, j]] ).powi(3);
                let i0_ldl = d_ij_3 - q_ij_3;

                // MDA for correlation integral without low-density limit                
                let i0_noldl = ((-eta_a[[i, j]]*0.5 + 1.0) / (-eta_a[[i, j]] + 1.0).powi(3) - 1.0) * (-q_ij_3 + rs_ij_3)
                                - ((-eta_b[[i, j]]*0.5 + 1.0) / (-eta_b[[i, j]] + 1.0).powi(3) - 1.0) * (-d_ij_3 + rs_ij_3) ;
                                
                // Helmholtz energy / (2/3 PI rho)
                a -= x[i] * x[j] * p.m[i] * p.m[j] * ( i0_noldl + i0_ldl );
                    
            }
        }        
        ( a * 2.0/3.0 * PI * state.partial_density.sum() ) * state.moles.sum() 
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{test_parameters, test_parameters_mixture};
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_delta_a0_wca_pure() {
        let moles = arr1(&[2.0]);

        // m = 12.0, t = 4.0, rho = 1.0

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let reduced_volume = moles[0] / reduced_density;

        let p = test_parameters(1.0, 24.0, 6.0, 1.0, 1.0);
        let pt = ReferencePerturbationWCA {
            parameters: Arc::new(p),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = pt.helmholtz_energy(&state) / moles[0];
        assert_relative_eq!(a, 0.258690311450425, epsilon = 1e-10);
    }
    #[test]
    fn test_delta_a0_wca_mixture() {
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

        let pt = ReferencePerturbationWCA {
            parameters: Arc::new(p),
        };
        let state = StateHD::new(reduced_temperature, reduced_volume, moles.clone());
        let a = pt.helmholtz_energy(&state) / (moles[0] + moles[1]);

        assert_relative_eq!(a, 0.308268896386771, epsilon = 1e-6);
    }
}
