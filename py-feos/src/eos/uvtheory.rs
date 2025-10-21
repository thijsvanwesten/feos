use super::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::parameter::PyParameters;
use crate::residual::ResidualModel;
use feos::uvtheory::{ChainContribution, CombinationRule, Perturbation, UVTheory, UVTheoryOptions};
use feos_core::{EquationOfState, ResidualDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pymethods]
impl PyEquationOfState {
    /// UV-Theory equation of state.
    ///
    /// Parameters
    /// ----------
    /// parameters : UVTheoryParameters
    ///     The parameters of the UV-theory equation of state to use.
    /// max_eta : float, optional
    ///     Maximum packing fraction. Defaults to 0.5.
    /// perturbation : "BH" | "WCA" | "WCA_B3" | "WCA_TPT", optional
    ///     Division type of the Mie potential. Defaults to "WCA".
    /// combination_rule : "arithmetic_phi" | "geometric_phi" | "geometric_psi" | "one_fluid_psi", optional
    ///     Rule used to combine parameters. Defaults to "geometric_psi".
    /// chain_contribution : "tpt1" | "tpt1y", optional
    ///     TPT version. Defaults to "tpt1y".
    /// max_iter_cross_assoc : unsigned integer, optional
    ///     Maximum number of iterations for cross association. Defaults to 50.
    /// tol_cross_assoc : float, optional
    ///     Tolerance for convergence of cross association. Defaults to 1e-10.
    ///
    /// Returns
    /// -------
    /// EquationOfState
    ///     The UV-Theory equation of state that can be used to compute thermodynamic
    ///     states.
    #[staticmethod]
    #[pyo3(
        signature = (parameters, max_eta=0.5, perturbation="WCA_TPT", combination_rule="geometric_psi", chain_contribution="tpt1y", max_iter_cross_assoc=50, tol_cross_assoc=1e-10),
        text_signature = r#"(parameters, max_eta=0.5, perturbation="WCA_TPT", combination_rule="geometric_psi", chain_contribution="tpt1y", max_iter_cross_assoc=50, tol_cross_assoc=1e-10)"#
    )]
    fn uvtheory(
        parameters: PyParameters,
        max_eta: f64,
        perturbation: &str,
        combination_rule: &str,
        chain_contribution: &str,
        max_iter_cross_assoc: usize,
        tol_cross_assoc: f64,
    ) -> PyResult<Self> {
        let perturbation = match perturbation {
            "BH" => Perturbation::BarkerHenderson,
            "WCA" => Perturbation::WeeksChandlerAndersen,
            "WCA_B3" => Perturbation::WeeksChandlerAndersenB3,
            "WCA_TPT" => Perturbation::WeeksChandlerAndersenTPT,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"perturbation must be "BH", "WCA", "WCA_B3" or "WCA_TPT""#.to_string(),
                ))
            }
        };
        let combination_rule = match combination_rule {
            "arithmetic_phi" => CombinationRule::ArithmeticPhi,
            "geometric_phi" => CombinationRule::GeometricPhi,
            "geometric_psi" => CombinationRule::GeometricPsi,
            "one_fluid_psi" => CombinationRule::OneFluidPsi,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"combination_rule must be "arithmetic_phi", "geometric_phi", "geometric_psi" or "one_fluid_psi""#.to_string(),
                ))
            }
        };
        let chain_contribution = match chain_contribution {
            "tpt1" => ChainContribution::TPT1,
            "tpt1y" => ChainContribution::TPT1y,
            _ => {
                return Err(PyErr::new::<PyValueError, _>(
                    r#"chain_contribution must be "tpt1" or "tpt1y""#.to_string(),
                ))
            }
        };
        let options = UVTheoryOptions {
            max_eta,
            perturbation,
            combination_rule,
            chain_contribution,
            max_iter_cross_assoc,
            tol_cross_assoc,
        };
        let residual =
            ResidualModel::UVTheory(UVTheory::with_options(parameters.try_convert()?, options));
        let ideal_gas = vec![IdealGasModel::NoModel; residual.components()];
        Ok(Self(Arc::new(EquationOfState::new(ideal_gas, residual))))
    }
}
