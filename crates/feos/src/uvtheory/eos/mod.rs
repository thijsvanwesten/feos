use crate::{
    association::{Association, AssociationStrength},
    hard_sphere::HardSphereProperties,
    uvtheory::{
        UVTheoryRecord,
        parameters::UVTheoryAssociationRecord,
        wca_tpt::{
            attractive_perturbation_wca::AttractivePerturbationWCA, chain_mie_tpty::ChainMie,
            hard_sphere_wca::HardSphereWCA, reference_perturbation_wca::ReferencePerturbationWCA,
        },
    },
};

use super::wca_tpt::chain_mie_tpty::gmie_aroundcontact_mix;
use super::wca_tpt::hard_sphere_wca::packing_fraction;

use super::parameters::{UVTheoryParameters, UVTheoryPars};
use feos_core::{Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::DualNum;
use quantity::MolarWeight;
use std::f64::consts::{FRAC_PI_6, PI};

// mod bh;
// pub use bh::BarkerHenderson;
mod wca;
pub use wca::{WeeksChandlerAndersen, WeeksChandlerAndersenB3};
pub mod wca_tpt;

const X_K21: [f64; 21] = [
    -0.995657163025808080735527280689003,
    -0.973906528517171720077964012084452,
    -0.930157491355708226001207180059508,
    -0.865063366688984510732096688423493,
    -0.780817726586416897063717578345042,
    -0.679409568299024406234327365114874,
    -0.562757134668604683339000099272694,
    -0.433395394129247190799265943165784,
    -0.294392862701460198131126603103866,
    -0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
    0.148874338981631210884826001129720,
    0.294392862701460198131126603103866,
    0.433395394129247190799265943165784,
    0.562757134668604683339000099272694,
    0.679409568299024406234327365114874,
    0.780817726586416897063717578345042,
    0.865063366688984510732096688423493,
    0.930157491355708226001207180059508,
    0.973906528517171720077964012084452,
    0.995657163025808080735527280689003,
];

const W_K21: [f64; 21] = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
    0.147739104901338491374841515972068,
    0.142775938577060080797094273138717,
    0.134709217311473325928054001771707,
    0.123491976262065851077958109831074,
    0.109387158802297641899210590325805,
    0.093125454583697605535065465083366,
    0.075039674810919952767043140916190,
    0.054755896574351996031381300244580,
    0.032558162307964727478818972459390,
    0.011694638867371874278064396062192,
];

/// Type of Combination Rule.
#[derive(Debug, Clone)]
pub enum CombinationRule {
    ArithmeticPhi,
    GeometricPhi,
    GeometricPsi,
    OneFluidPsi,
}

/// Type of perturbation.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Perturbation {
    BarkerHenderson,
    WeeksChandlerAndersen,
    WeeksChandlerAndersenB3,
    WeeksChandlerAndersenTPT,
}

#[derive(Clone)]
pub enum ChainContribution {
    TPT1,
    TPT1y,
}

#[derive(Debug, Clone, Copy)]
pub enum AssociationModel {
    TVW,
    Lafitte,
}

/// Configuration options for uv-theory
#[derive(Clone)]
pub struct UVTheoryOptions {
    pub max_eta: f64,
    pub perturbation: Perturbation,
    pub combination_rule: CombinationRule,
    pub chain_contribution: ChainContribution,
    pub association_model: AssociationModel,
    pub max_iter_cross_assoc: usize,
    pub tol_cross_assoc: f64,
}

impl Default for UVTheoryOptions {
    fn default() -> Self {
        Self {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersenTPT,
            combination_rule: CombinationRule::OneFluidPsi,
            chain_contribution: ChainContribution::TPT1y,
            association_model: AssociationModel::TVW,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        }
    }
}

/// uv-theory equation of state
pub struct UVTheory {
    parameters: UVTheoryParameters,
    params: UVTheoryPars,
    options: UVTheoryOptions,
    association: Option<Association<UVTheoryPars>>,
}

impl UVTheory {
    /// uv-theory with default options (WCA).
    pub fn new(parameters: UVTheoryParameters) -> Self {
        Self::with_options(parameters, UVTheoryOptions::default())
    }

    /// uv-theory with provided options.
    pub fn with_options(parameters: UVTheoryParameters, options: UVTheoryOptions) -> Self {
        let params =
            UVTheoryPars::new(&parameters, options.perturbation, options.association_model);

        let association = Association::new(
            &parameters,
            options.max_iter_cross_assoc,
            options.tol_cross_assoc,
        )
        .unwrap();

        Self {
            parameters,
            params,
            options,
            association,
        }
    }

    pub fn reduced_helmholtz_energy_density_contributions_wca_tpt<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        let mut contributions = vec![
            (
                "Hard Sphere (WCA, TPT)",
                HardSphereWCA.helmholtz_energy_density(&self.params, state),
            ),
            (
                "Mie Chain",
                ChainMie {
                    chain_contribution: self.options.chain_contribution.clone(),
                }
                .helmholtz_energy_density(&self.params, state),
            ),
            (
                "Reference Perturbation (WCA)",
                ReferencePerturbationWCA.helmholtz_energy_density(&self.params, state),
            ),
            (
                "Attractive Perturbation (WCA)",
                AttractivePerturbationWCA.helmholtz_energy_density(&self.params, state),
            ),
        ];
        if let Some(association) = self.association.as_ref() {
            let d = self.params.hs_diameter(state.temperature);
            contributions.push((
                "Association",
                association.helmholtz_energy_density(
                    &self.params,
                    &self.parameters.association,
                    state,
                    &d,
                ),
            ))
        }
        contributions
    }
}

impl Subset for UVTheory {
    fn subset(&self, component_list: &[usize]) -> Self {
        Self::with_options(self.parameters.subset(component_list), self.options.clone())
    }
}

impl ResidualDyn for UVTheory {
    fn components(&self) -> usize {
        self.parameters.pure.len()
    }

    fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
        let msigma3 = self
            .params
            .m
            .component_mul(&self.params.sigma.map(|v| v.powi(3)));
        (msigma3.map(D::from).dot(molefracs) * FRAC_PI_6).recip() * self.options.max_eta
    }

    fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
    ) -> Vec<(&'static str, D)> {
        match &self.options.perturbation {
            Perturbation::BarkerHenderson => {
                todo!()
                // BarkerHenderson.residual_helmholtz_energy_contributions(&self.params, state)
            }
            Perturbation::WeeksChandlerAndersen => {
                WeeksChandlerAndersen.residual_helmholtz_energy_contributions(&self.params, state)
            }
            Perturbation::WeeksChandlerAndersenB3 => {
                WeeksChandlerAndersenB3.residual_helmholtz_energy_contributions(&self.params, state)
            }
            Perturbation::WeeksChandlerAndersenTPT => {
                self.reduced_helmholtz_energy_density_contributions_wca_tpt(state)
            }
        }
    }
}

impl UVTheoryPars {
    fn association_strength_lafitte<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
        diameter: &DVector<D>,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &UVTheoryAssociationRecord,
    ) -> D {
        // auxiliary variables
        let [zeta2, n3] = self.zeta(state.temperature, &state.partial_density, [2, 3]);
        let n2 = zeta2 * 6.0;
        let n3i = (-n3 + 1.0).recip();

        let di = diameter[comp_i];
        let dj = diameter[comp_j];
        let k = di * dj / (di + dj) * (n2 * n3i);
        let g_contact = n3i * (k * (k / 18.0 + 0.5) + 1.0);
        
        let d = (di + dj) * 0.5;
        // temperature dependent association volume
        // rc and rd are dimensioned in units of Angstrom
        let rc = assoc_ij.rc_ab;
        // rd is the distance between an association site and the segment centre.
        // It is fixed at 0.4 sigma, leading to 0.4 * 0.5 = 0.2 in the combining rule.
        let rd = (self.sigma[comp_i] + self.sigma[comp_j]) * 0.2;
        let v = d * d * PI * 4.0 / (72.0 * rd.powi(2))
            * ((d.recip() * (rc + 2.0 * rd)).ln()
                * (6.0 * rc.powi(3) + 18.0 * rc.powi(2) * rd - 24.0 * rd.powi(3))
                + (-d + rc + 2.0 * rd)
                    * (d.powi(2) + d * rc + 22.0 * rd.powi(2)
                        - 5.0 * rc * rd
                        - d * 7.0 * rd
                        - 8.0 * rc.powi(2)));
        let f_ab = (state.temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1();
        
        g_contact * f_ab * v
    }

    fn association_strength_tvw<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
        diameter: &DVector<D>,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &UVTheoryAssociationRecord,
    ) -> D {
        let n = diameter.len();
        let rho_st = (0..n).fold(D::zero(), |z, i| {
            z + state.partial_density[i] * self.m[i] * self.sigma[i].powi(3)
        });
        let sigma_ij = self.sigma[(comp_i, comp_j)];
        let sigma_ij_inv = 1.0 / sigma_ij;
        let eta = packing_fraction(&self.m, &state.partial_density, diameter);

        // geometry of SW assocation site: rc and rd are dimensioned in units of Angstrom
        let rc = assoc_ij.rc_ab;
        let rd = assoc_ij.rd_ab;

        // 20-point gauss-legendre integration of association integral delta
        let rmin = 0.8 * sigma_ij;
        let rmax = 2.0 * rd + rc;
        let width = (rmax - rmin) * 0.5;

        let two_rd = 2.0 * rd;
        let fac = width + rmin;
        let fac1 = two_rd + rc;
        let fac2 = 2.0 * rc - two_rd;

        let mut i_ab_ij = D::zero();

        for k in 0..21 {
            let r = width * X_K21[k] + fac;
            let r_geometry = (fac1 - r).powi(2) * (fac2 + r);
            let integrand = gmie_aroundcontact_mix(
                r * sigma_ij_inv,
                self,
                eta,
                &state.partial_density,
                state.temperature,
                diameter,
                &rho_st,
                comp_i,
                comp_j,
            ) * r_geometry
                * r;
            i_ab_ij += integrand * width * W_K21[k];
        }
        // Notes: improve efficiency by only callling y^hs in integrand? optimize rmin to reduce to n<20-point GL?
        i_ab_ij = i_ab_ij * 4.0 * PI / (24.0 * rd * rd);
        let f_ab = (state.temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1();
        let delta = i_ab_ij * f_ab;

        // dbg!(rc, rd, rmin, rmax, sigma_ij, &state.partial_density, state.temperature, rho_st, width,i_ab_ij, f_ab, delta);
        delta
    }
}

impl AssociationStrength for UVTheoryPars {
    type Pure = UVTheoryRecord;
    type Record = UVTheoryAssociationRecord;

    // fn association_strength<D: DualNum<f64> + Copy>(
    //     &self,
    //     state: &feos_core::StateHD<D>,
    //     diameter: &DVector<D>,
    //     comp_i: usize,
    //     comp_j: usize,
    //     assoc_ij: &Self::Record,
    // ) -> D {
    //     let [zeta2, n3] = self.zeta(state.temperature, &state.partial_density, [2, 3]);
    //     let n2 = zeta2 * 6.0;
    //     let n3i = (-n3 + 1.0).recip();
    //     let di = diameter[comp_i];
    //     let dj = diameter[comp_j];
    //     let k = di * dj / (di + dj) * (n2 * n3i);
    //     let g_contact = n3i * (k * (k / 18.0 + 0.5) + 1.0);

    //     let d = (di + dj) * 0.5;

    //     // temperature dependent association volume
    //     // rc and rd are dimensioned in units of Angstrom
    //     let rc = assoc_ij.rc_ab;
    //     let rd = assoc_ij.rd_ab;

    //     let k_ab_ij = d * d * PI * 4.0 / (72.0 * rd.powi(2))
    //         * ((d.recip() * (rc + 2.0 * rd)).ln()
    //             * (6.0 * rc.powi(3) + 18.0 * rc.powi(2) * rd - 24.0 * rd.powi(3))
    //             + (-d + rc + 2.0 * rd)
    //                 * (d.powi(2) + d * rc + 22.0 * rd.powi(2)
    //                     - 5.0 * rc * rd
    //                     - d * 7.0 * rd
    //                     - 8.0 * rc.powi(2)));
    //     let i_ab_ij = g_contact * k_ab_ij;
    //     let delta = i_ab_ij * (state.temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1();
    //     dbg!(rc, rd, g_contact, k_ab_ij, i_ab_ij, delta);
    //     delta
    // }

    fn association_strength<D: DualNum<f64> + Copy>(
        &self,
        state: &feos_core::StateHD<D>,
        diameter: &DVector<D>,
        comp_i: usize,
        comp_j: usize,
        assoc_ij: &Self::Record,
    ) -> D {
        match self.association_model {
            AssociationModel::Lafitte => {
                self.association_strength_lafitte(state, diameter, comp_i, comp_j, assoc_ij)
            }
            AssociationModel::TVW => {
                self.association_strength_tvw(state, diameter, comp_i, comp_j, assoc_ij)
            }
        }

        // let n = diameter.len();
        // let rho_st = (0..n).fold(D::zero(), |z, i| z + state.partial_density[i] * self.m[i] * self.sigma[i].powi(3));
        // let sigma_ij = self.sigma[(comp_i,comp_j)];
        // let sigma_ij_inv = 1.0/sigma_ij;
        // let eta = packing_fraction(&self.m, &state.partial_density, diameter);

        // // geometry of SW assocation site: rc and rd are dimensioned in units of Angstrom
        // let rc = assoc_ij.rc_ab;
        // let rd = assoc_ij.rd_ab;

        // // 20-point gauss-legendre integration of association integral delta
        // let rmin = 0.8*sigma_ij;
        // let rmax = 2.0*rd + rc;
        // let width = (rmax-rmin)*0.5;

        // let two_rd = 2.0*rd;
        // let fac = width + rmin;
        // let fac1 = two_rd + rc;
        // let fac2 = 2.0*rc - two_rd;

        // let mut i_ab_ij = D::zero();

        // for k in 0..21 {
        //     let r = width * X_K21[k] + fac;
        //     let r_geometry = (fac1 - r).powi(2)*(fac2 + r);
        //     let integrand = gmie_aroundcontact_mix(
        //         r*sigma_ij_inv,
        //         self,
        //         eta,
        //         &state.partial_density,
        //         state.temperature,
        //         diameter,
        //         &rho_st,
        //         comp_i,
        //         comp_j
        //     ) * r_geometry *r;
        //     i_ab_ij += integrand * width * W_K21[k];
        // }
        // // Notes: improve efficiency by only callling y^hs in integrand? optimize rmin to reduce to n<20-point GL?
        // i_ab_ij = i_ab_ij * 4.0*PI / (24.0*rd*rd);
        // let f_ab = (state.temperature.recip() * assoc_ij.epsilon_k_ab).exp_m1();
        // let delta = i_ab_ij * f_ab;

        // // dbg!(rc, rd, rmin, rmax, sigma_ij, &state.partial_density, state.temperature, rho_st, width,i_ab_ij, f_ab, delta);
        // delta
    }

    fn combining_rule(
        pure_i: &Self::Pure,
        pure_j: &Self::Pure,
        parameters_i: &Self::Record,
        parameters_j: &Self::Record,
    ) -> Self::Record {
        let rc_ab = (parameters_i.rc_ab * pure_i.sigma + parameters_j.rc_ab * pure_j.sigma) * 0.5;
        let rd_ab = (parameters_i.rd_ab * pure_i.sigma + parameters_j.rd_ab * pure_j.sigma) * 0.5;
        // geometric (SAFT-VR Mie)
        let epsilon_k_ab = (parameters_i.epsilon_k_ab * parameters_j.epsilon_k_ab).sqrt();
        // arithmetic (PC-SAFT)
        // let epsilon_k_ab = 0.5 * (parameters_i.epsilon_k_ab + parameters_j.epsilon_k_ab);
        Self::Record {
            rc_ab,
            rd_ab,
            epsilon_k_ab,
        }
    }
}

impl Molarweight for UVTheory {
    fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
        self.parameters.molar_weight.clone()
    }
}

#[cfg(test)]
#[expect(clippy::excessive_precision)]
mod test {
    use super::*;
    use crate::uvtheory::parameters::utils::{new_simple, test_parameters_mixture};
    use crate::uvtheory::parameters::*;
    use approx::assert_relative_eq;
    use feos_core::parameter::{Identifier, PureRecord};
    use feos_core::{FeosResult, State};
    use nalgebra::dvector;
    use quantity::{ANGSTROM, KELVIN, MOL, NAV, RGAS};
    use typenum::P3;

    #[test]
    fn helmholtz_energy_pure_wca() -> FeosResult<()> {
        let sig = 3.7039;
        let eps_k = 150.03;
        let parameters = new_simple(1.0, 24.0, 6.0, sig, eps_k);
        let eos = &UVTheory::new(parameters);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        assert_relative_eq!(a, 2.972986567516, max_relative = 1e-12); //wca
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_bh() -> FeosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 24.0;
        let att = 6.0;
        let parameters = new_simple(1.0, rep, att, sig, eps_k);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
            combination_rule: CombinationRule::OneFluidPsi,
            chain_contribution: ChainContribution::TPT1y,
            association_model: AssociationModel::TVW,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        };
        let eos = &UVTheory::with_options(parameters, options);

        let reduced_temperature = 4.0;
        let reduced_density = 1.0;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();

        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();

        assert_relative_eq!(a, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_uvb3() -> FeosResult<()> {
        let eps_k = 150.03;
        let sig = 3.7039;
        let rep = 12.0;
        let att = 6.0;
        let parameters = new_simple(1.0, rep, att, sig, eps_k);
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersenB3,
            combination_rule: CombinationRule::OneFluidPsi,
            chain_contribution: ChainContribution::TPT1y,
            association_model: AssociationModel::TVW,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        };
        let eos = &UVTheory::with_options(parameters, options);

        let reduced_temperature = 4.0;
        let reduced_density = 0.5;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_molar_helmholtz_energy() / (RGAS * temperature)).into_value();
        dbg!(a);
        assert_relative_eq!(a, 0.37659379124271003, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_mixtures_bh() -> FeosResult<()> {
        // Mixture of equal components --> result must be the same as for pure fluid ///
        // component 1
        let rep1 = 24.0;
        let eps_k1 = 150.03;
        let sig1 = 3.7039;
        let r1 = UVTheoryRecord::new(1.0, rep1, 6.0, sig1, eps_k1);
        let i = Identifier::new(None, None, None, None, None, None);
        // compontent 2
        let rep2 = 24.0;
        let eps_k2 = 150.03;
        let sig2 = 3.7039;
        let r2 = UVTheoryRecord::new(1.0, rep2, 6.0, sig2, eps_k2);
        let j = Identifier::new(None, None, None, None, None, None);
        //////////////

        let pr1 = PureRecord::new(i, 1.0, r1);
        let pr2 = PureRecord::new(j, 1.0, r2);
        let uv_parameters = UVTheoryParameters::new_binary([pr1, pr2], None, vec![])?;
        // state
        let reduced_temperature = 4.0;
        let eps_k_x = (eps_k1 + eps_k2) / 2.0; // Check rule!!
        let t_x = reduced_temperature * eps_k_x * KELVIN;
        let sig_x = (sig1 + sig2) / 2.0; // Check rule!!
        let reduced_density = 1.0;
        let moles = dvector![1.7, 0.3] * MOL;
        let total_moles = moles.sum();
        let volume = (sig_x * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::BarkerHenderson,
            combination_rule: CombinationRule::OneFluidPsi,
            chain_contribution: ChainContribution::TPT1y,
            association_model: AssociationModel::TVW,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        };

        let eos_bh = &UVTheory::with_options(uv_parameters, options);

        let state_bh = State::new_nvt(&eos_bh, t_x, volume, &moles).unwrap();
        let a_bh = (state_bh.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();

        assert_relative_eq!(a_bh, 2.993577305779432, max_relative = 1e-12);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture() -> FeosResult<()> {
        let parameters = test_parameters_mixture(
            dvector![1.0, 1.0],
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 1.0],
            dvector![1.0, 0.5],
        );
        let p = UVTheoryPars::new(
            &parameters,
            Perturbation::WeeksChandlerAndersen,
            AssociationModel::TVW,
        );

        // state
        let reduced_temperature = 1.0;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let reduced_density = 0.9;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = (p.sigma[0] * ANGSTROM).powi::<P3>() / reduced_density * NAV * total_moles;

        // EoS
        let eos_wca = &UVTheory::new(parameters);
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_helmholtz_energy() / (RGAS * t_x * state_wca.total_moles))
            .into_value();

        assert_relative_eq!(a_wca, -0.597791038364405, max_relative = 1e-5);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_wca_mixture_different_sigma() -> FeosResult<()> {
        let parameters = test_parameters_mixture(
            dvector![1.0, 1.0],
            dvector![12.0, 12.0],
            dvector![6.0, 6.0],
            dvector![1.0, 2.0],
            dvector![1.0, 0.5],
        );
        let p = UVTheoryPars::new(
            &parameters,
            Perturbation::WeeksChandlerAndersen,
            AssociationModel::TVW,
        );

        // state
        let reduced_temperature = 1.5;
        let t_x = reduced_temperature * p.epsilon_k[0] * KELVIN;
        let sigma_x_3 = (0.4 + 0.6 * 8.0) * ANGSTROM.powi::<P3>();
        let density = 0.52000000000000002 / sigma_x_3;
        let moles = dvector![0.4, 0.6] * MOL;
        let total_moles = moles.sum();
        let volume = NAV * total_moles / density;

        // EoS
        let eos_wca = &UVTheory::new(parameters);
        let state_wca = State::new_nvt(&eos_wca, t_x, volume, &moles).unwrap();
        let a_wca = (state_wca.residual_molar_helmholtz_energy() / (RGAS * t_x)).into_value();
        assert_relative_eq!(a_wca, -0.034206207363139396, max_relative = 1e-5);
        Ok(())
    }

    #[test]
    fn helmholtz_energy_pure_miechain() -> FeosResult<()> {
        let sig = 1.0;
        let eps_k = 1.0;
        let parameters = new_simple(5.0, 24.0, 6.0, sig, eps_k);

        let options = UVTheoryOptions {
            max_eta: 0.5,
            perturbation: Perturbation::WeeksChandlerAndersenTPT,
            combination_rule: CombinationRule::OneFluidPsi,
            chain_contribution: ChainContribution::TPT1y,
            association_model: AssociationModel::TVW,
            max_iter_cross_assoc: 50,
            tol_cross_assoc: 1e-10,
        };
        let eos = &UVTheory::with_options(parameters, options);

        let reduced_temperature = 2.0;
        let reduced_density = 0.15;
        let temperature = reduced_temperature * eps_k * KELVIN;
        let moles = dvector![2.0] * MOL;
        let volume = (sig * ANGSTROM).powi::<P3>() / reduced_density * NAV * 2.0 * MOL;
        let s = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
        let a = (s.residual_helmholtz_energy() / (s.total_moles * RGAS * temperature)).into_value();

        let contributions = s.residual_molar_helmholtz_energy_contributions();

        for (name, value) in contributions.iter() {
            let a_red = (value / (RGAS * s.temperature)).into_value();
            println!("{:<30}: A / NkT = {:>.10}", &name, a_red);
        }

        assert_relative_eq!(a, 0.93410937628984314, max_relative = 1e-9);
        // assert_relative_eq!(a, 0.18900517901738298, max_relative = 1e-12);
        Ok(())
    }
}
