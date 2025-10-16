use feos_core::StateHD;
use num_dual::DualNum;

use crate::uvtheory::{eos::wca_tpt::hard_sphere_wca::HardSphereWCA, parameters::UVTheoryPars};

pub mod hard_sphere_wca;
pub mod chain_mie_tpty;

pub struct WeeksChandlerAndersenTPT;

impl WeeksChandlerAndersenTPT {
    pub fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy>(
        &self,
        parameters: &UVTheoryPars,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        vec![
            (
                "Hard Sphere (WCA, TPT)",
                HardSphereWCA.helmholtz_energy_density(parameters, state),
            ),
            (
                "Mie Chain",
                HardSphereWCA.helmholtz_energy_density(parameters, state),
            ),
            // (
            //     "Reference Perturbation (WCA)",
            //     ReferencePerturbation.helmholtz_energy_density(parameters, state),
            // ),
            // (
            //     "Attractive Perturbation (WCA)",
            //     AttractivePerturbation.helmholtz_energy_density(parameters, state),
            // ),
        ]
    }
}
