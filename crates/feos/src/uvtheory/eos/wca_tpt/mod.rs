use feos_core::StateHD;
use num_dual::DualNum;

use crate::uvtheory::{eos::wca_tpt::hard_sphere_wca::HardSphereWCA, parameters::UVTheoryPars};
use crate::uvtheory::{eos::wca_tpt::reference_perturbation_wca::ReferencePerturbationWCA};
use crate::uvtheory::{eos::wca_tpt::attractive_perturbation_wca::AttractivePerturbationWCA};
use crate::uvtheory::{eos::wca_tpt::chain_mie_tpty::ChainMie};
use crate::uvtheory::eos::ChainContribution;

pub mod hard_sphere_wca;
pub mod chain_mie_tpty;
pub mod reference_perturbation_wca;
pub mod attractive_perturbation_wca;

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
                ChainMie{chain_contribution:ChainContribution::TPT1y}.helmholtz_energy_density(parameters, state),
            ),
            (
                "Reference Perturbation (WCA)",
                ReferencePerturbationWCA.helmholtz_energy_density(parameters, state),
            ),
            (
                "Attractive Perturbation (WCA)",
                AttractivePerturbationWCA.helmholtz_energy_density(parameters, state),
            ),
        ]
    }
}
