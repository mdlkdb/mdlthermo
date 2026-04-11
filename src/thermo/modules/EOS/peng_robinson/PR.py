import json
from os.path import join as opj
from pathlib import Path

import numpy as np
import scipy.optimize as spo
from rdkit import Chem

from COSMOSAC import COSMOSAC
from GCGCN import predict_TC, predict_PC, predict_omega


FILE_DIR = Path(__file__).parent


class Peng_Robinson:
    """
    Peng-Robinson equation of state
    """

    def __init__(self):
        self.R = 83.1446261815324  # Gas constant, cm3*bar/K/mol

        # PR EoS constants
        self.eta_c = (1 + (4 - 8**0.5) ** (1 / 3) + (4 + 8**0.5) ** (1 / 3)) ** (-1)
        self.Omega_a = (8 + 40 * self.eta_c) / (49 - 37 * self.eta_c)
        self.Omega_b = self.eta_c / (3 + self.eta_c)

        # PR EoS parameters
        self.ac = np.array([])  # PR EoS ac parameter
        self.b = np.array([])  # PR EoS b parameter

        # Component information
        self.smiles = []  # SMILES expression
        self.Tc = np.array([])  # Critical temperature, [K]
        self.pc = np.array([])  # Critical pressure, [bar]
        self.omega = np.array([])  # Acentric factor, [1]
        self.c = np.array([]).reshape(0, 3)  # M.-C. alpha function parameters

        # Critical point information
        self.Zc = (11 - 2 * 7**0.5 * np.sinh(np.arcsinh(13 / 7 / 7**0.5) / 3)) / 32

        # COSMO-SAC-ML model
        self.model = COSMOSAC(version=2010, predict=True)

    def cal_ac(self, Tc, pc):
        """Calculate PR EoS ac parameter.

        Parameters
        ----------
        Tc : float
            Critical temperature [K]
        pc : float
            Critical pressure [bar]

        Returns
        -------
        float
            PR EoS ac parameter [bar*cm^6/mol^2]
        """
        return self.Omega_a * self.R**2 * Tc**2 / pc

    def cal_b(self, Tc, pc):
        """Calculate PR EoS b parameter.

        Parameters
        ----------
        Tc : float
            Critical temperature [K]
        pc : float
            Critical pressure [bar]

        Returns
        -------
        float
            PR EoS b parameter [cm^3/mol]
        """
        return self.Omega_b * self.R * Tc / pc

    def add_comp(self, smiles, Tc=None, pc=None, omega=None, c=None):
        """Add component to PR EoS.

        Parameters
        ----------
        smiles : str
            SMILES expression of component
        Tc : float, optional
            Critical temperature [K]
        pc : float, optional
            Critical pressure [bar]
        omega : float, optional
            Acentric factor [1]
        c : list, optional
            Mathias-Copeman alpha function parameters [1]

        Returns
        -------
        None
            Updates properties of each component.
        """
        # Load pre-defined parameters
        with open(opj(FILE_DIR, "pr_param.json"), mode="r") as f:
            pr_param = json.load(f)

        inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))

        # Define helper function to get property value
        def get_property(prop, key, pred_func, default=None):
            if prop is not None:
                return prop
            elif inchikey in pr_param and key in pr_param[inchikey]:
                return pr_param[inchikey][key]
            else:
                return pred_func(smiles) if callable(pred_func) else default

        # Resolve properties
        Tc = get_property(Tc, "Tc", lambda s: predict_TC(s)[0])
        pc = get_property(pc, "pc", lambda s: predict_PC(s)[0] / 100)
        omega = get_property(omega, "omega", predict_omega)
        c = get_property(c, "c", None, np.array([np.nan] * 3))

        # Save the inputs
        self.smiles.append(smiles)
        self.Tc = np.append(self.Tc, Tc)
        self.pc = np.append(self.pc, pc)
        self.omega = np.append(self.omega, omega)
        self.c = np.vstack([self.c, c])

        # Calculate dependent properties
        self.ac = np.append(self.ac, self.cal_ac(Tc, pc))
        self.b = np.append(self.b, self.cal_b(Tc, pc))

        # Update model
        self.model.add_comp(SMILES=smiles)

    def alpha_pr(self, T, Tc, omega):
        """Calculate PR EoS alpha parameter by Peng-Robinson 1980.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K]
        Tc : float or numpy.ndarray
            Critical temperature [K]
        omega : float or numpy.ndarray
            Acentric factor [1]

        Returns
        -------
        float or numpy.ndarray
            PR EoS alpha parameter [1]
        """
        Tr = T / Tc  # Reduced temperature

        m = np.where(
            omega <= 0.491,
            0.37464 + 1.54226 * omega - 0.26992 * omega**2,
            0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3,
        )

        return (1 + m * (1 - Tr**0.5)) ** 2

    def alpha_mc(self, T, Tc, c1, c2, c3):
        """Calculate PR EoS alpha parameter by Mathias-Copeman 1983.

        Parameters
        ----------
        T : float or numpy.ndarray
            Temperature [K]
        Tc : float or numpy.ndarray
            Critical temperature [K]
        c1, c2, c3 : float or numpy.ndarray
            Mathias-Copeman alpha function parameter.

        Returns
        -------
        float or numpy.ndarray
            PR EoS alpha parameter [1]
        """
        Tr = T / Tc  # Reduced temperature
        term = 1 - Tr**0.5

        alpha = np.where(
            Tr <= 1,
            (1 + c1 * term + c2 * term**2 + c3 * term**3) ** 2,
            (1 + c1 * term) ** 2,
        )
        return alpha

    def mix_mhv2(self, smiles, T, x, a, b):
        """Mix components using the modified Huron-Vidal mixing rule 2.

        Parameters
        ----------
        smiles : list
            List of SMILES strings for components.
        T : float
            Temperature of the system, in K.
        x : numpy.ndarray
            Mole fractions of each component.

        Returns
        -------
        float
            Mixture 'a' parameter [bar*cm^6/mol^2]
        float
            Mixture 'b' parameter [cm^3/mol]
        """
        # Calculate excess Gibbs energy
        self.model.x = x
        self.model.T = T
        gE = np.sum(x * np.log(self.model.gam()))  # G^{E}/RT

        # Calculate mixture b
        b_mix = np.sum(x * b)

        # Calculate mixture a
        alpha = a / (b * self.R * T)
        sum_alpha = np.sum(x * alpha)
        sum_alpha2 = np.sum(x * alpha**2)

        q1 = -0.4347
        q2 = -0.003654
        term = -q1 * sum_alpha - q2 * sum_alpha2 - gE - np.sum(x * np.log(b_mix / b))

        alphamix = np.roots([q2, q1, term])
        alphamix = np.max(alphamix)  # Reliable range: 11.7-16.7

        a_mix = alphamix * b_mix * self.R * T

        return a_mix, b_mix

    def cal_crit_mix(self, x):
        """Calculate critical properties of mixture.

        Parameters
        ----------
        x : numpy.ndarray
            Mole fractions of each component.

        Returns
        -------
        None
            Updates critical properties of mixture
            (pcmix, Vcmix, Tcmix, acmix).
        """
        Omega_a = self.Omega_a
        Omega_b = self.Omega_b

        # Calculate critical temperature
        Tcmix = np.mean(self.Tc)  # initial guess
        Tcmix_old = 1

        # Calculate bmix
        bmix = np.sum(x * self.b)

        for _ in range(50):
            # Stop iteration if Tcmix converged
            if abs((Tcmix - Tcmix_old) / Tcmix_old) <= 1e-4:
                break

            Tcmix_old = Tcmix

            # (1) acmix and bmix are calculated from Tcmix
            acmix, bmix = self.mix_mhv2(self.smiles, Tcmix, x, self.ac, self.b)

            # (2) The ratio of acmix and bmix yields Tcmix
            Tcmix = (acmix * Omega_b) / (Omega_a * bmix * self.R)

        # Calculate critical volume
        Vcmix = bmix * self.Zc / Omega_b

        # Calculate critical pressure
        pcmix = (acmix * Omega_b**2) / (Omega_a * bmix**2)

        # Save the results
        self.pcmix = pcmix
        self.Vcmix = Vcmix
        self.Tcmix = Tcmix
        self.acmix = acmix

    def set_state(self, T, x):
        """Set state of PR EoS.

        Parameters
        ----------
        T : float
            Temperature [K]
        x : numpy.ndarray
            Mole fractions of each component

        Returns
        -------
        None
            Updates state of PR EoS.
        """
        # Calculate alpha function
        alpha_mc = self.alpha_mc(T, self.Tc, self.c[:, 0], self.c[:, 1], self.c[:, 2])
        alpha_pr = self.alpha_pr(T, self.Tc, self.omega)
        alpha = np.where(np.isnan(alpha_mc), alpha_pr, alpha_mc)

        # Calculate a for each component
        a = self.ac * alpha

        # Calculate mixture a, b
        amix, bmix = self.mix_mhv2(self.smiles, T, x, a, self.b)

        # Calculate critical properties of mixture
        self.cal_crit_mix(x)

        # Save the mixing results
        self.T = T
        self.x = x

        self.a = a
        self.amix = amix
        self.bmix = bmix

        self.model.x = x
        self.model.T = T

    # =========================================================================
    # These functions under this line should be used after using set_state.
    # =========================================================================

    def p(self, rho):
        """Calculate pressure from given density.

        Parameters
        ----------
        rho : float
            Density [mol/cm^3]

        Returns
        -------
        float
            Pressure [bar]
        """
        # Define system parameters
        a = self.amix
        b = self.bmix
        R = self.R
        T = self.T

        return rho * R * T / (1 - b * rho) - a * rho**2 / (
            1 + 2 * b * rho - b**2 * rho**2
        )

    def dp(self, rho):
        """Calculate pressure derivative with respect to density.

        Parameters
        ----------
        rho : float
            Density [mol/cm3]

        Returns
        -------
        float
            Pressure derivative [bar*cm^3/mol]
        """
        # Define system parameters
        a = self.amix
        b = self.bmix
        R = self.R
        T = self.T

        # Calculate pressure derivative
        term = 2 * a * (b * rho**2 + rho) / (1 + 2 * b * rho - b**2 * rho**2) ** 2
        return R * T / (1 - b * rho) ** 2 - term

    def obj_dp(self, rho):
        """Calculate objective function for finding boundary density.

        Parameters
        ----------
        rho : float
            Density [mol/cm3]

        Returns
        -------
        float
            Squared difference between pressure derivative and threshold
        """
        return (self.dp(rho) - 0.1 * self.R * self.T) ** 2

    def rho_EoS(self, p):
        """Calculate vapor and liquid densities from PR EoS.

        Parameters
        ----------
        p : float
            Pressure [bar]

        Returns
        -------
        tuple of float
            Vapor and liquid densities [mol/cm^3]
        """
        a = self.amix
        b = self.bmix
        R = self.R
        T = self.T

        b_lo = (1 - 2**0.5) / b
        b_hi = 1 / b

        coeff_0 = -p
        coeff_1 = R * T - b * p
        coeff_2 = 2 * b * R * T + 3 * b**2 * p - a
        coeff_3 = a * b - b**3 * p - b**2 * R * T

        rho_EoS = np.roots([coeff_3, coeff_2, coeff_1, coeff_0])
        rho_EoS = rho_EoS[np.isreal(rho_EoS)].real  # Real number
        rho_EoS = rho_EoS[(rho_EoS > b_lo) & (rho_EoS < b_hi)]  # Valid rho

        return rho_EoS.min(), rho_EoS.max()

    def rho(self, p):
        """Calculate vapor and liquid densities with density extrapolation.

        Parameters
        ----------
        p : float
            Pressure [bar]

        Returns
        -------
        tuple of float
            Vapor and liquid densities [mol/cm^3]
        """
        b = self.bmix
        R = self.R
        T = self.T

        # 1. Calculate basic densities
        rho_lo = (1 - 2**0.5) / b
        rho_hi = 1 / b
        rho_mc = (8**0.5 * np.sinh(np.arcsinh(8**0.5) / 3) - 1) / (3 * b)

        rho_vap_EoS, rho_liq_EoS = self.rho_EoS(p)

        # 2. Calculate liquid density
        # Calculate boundary density
        solve_liq_omega = spo.fsolve(self.obj_dp, rho_hi - 0.1 / b, full_output=True)

        if np.isclose(solve_liq_omega[1]["fvec"][0], 0):  # solution found
            rho_liq_omega = solve_liq_omega[0][0]
            rho_liq_bound = np.median([rho_mc, rho_liq_omega, rho_hi])
        else:  # solution not found
            rho_liq_bound = rho_mc

        # Calculate extrapolated density
        p_bound = self.p(rho_liq_bound)
        dp_bound = self.dp(rho_liq_bound)

        B = dp_bound * (rho_liq_bound - 0.7 * rho_mc)
        A = p_bound - B * np.log(rho_liq_bound - 0.7 * rho_mc)

        rho_liq_extrap = np.min([np.exp((p - A) / B) + 0.7 * rho_mc, rho_hi])

        # Select liquid density
        rho_liq = np.median([rho_liq_EoS, rho_liq_bound, rho_liq_extrap])

        # 3. Calculate vapor density
        # Calculate boundary density
        solve_vap_omega = spo.fsolve(self.obj_dp, rho_lo + 0.1 / b, full_output=True)

        if np.isclose(solve_vap_omega[1]["fvec"][0], 0):  # solution found
            rho_vap_omega = solve_vap_omega[0][0]
            rho_vap_bound = np.median([rho_lo, rho_vap_omega, 0.9 * rho_mc])
        else:  # solution not found
            rho_vap_bound = 0.9 * rho_mc

        # Calculate extrapolated density
        p_bound = self.p(rho_vap_bound)
        dp_bound = self.dp(rho_vap_bound)

        A = 1 / p_bound
        B = -dp_bound / p_bound**2
        C_rho = 0.5 * (rho_mc - rho_vap_bound)
        C = -abs(A + B * C_rho) / C_rho**2

        term_1 = np.min([0, p - p_bound]) / (0.1 * R * T)
        term_2 = (-B - np.sqrt(B**2 - 4 * C * np.max([0, A - 1 / p]))) / (2 * C)
        term_31 = np.max([0, T - self.Tcmix])
        term_32 = np.max([0, self.dp(rho_vap_bound) - 0.1 * R * T])
        term_33 = rho_vap_EoS - rho_vap_bound

        term = rho_vap_bound + term_1 + term_2 + term_31 * term_32 * term_33
        rho_vap_extrap = np.median([0, rho_hi, term])

        # Select vapor density
        rho_vap = np.median([rho_vap_EoS, rho_vap_bound, rho_vap_extrap])

        return rho_vap, rho_liq

    def phi_hat(self, p):
        """Calculate partial fugacity coefficients for vapor and liquid phases.

        Parameters
        ----------
        p : float
            Pressure [bar]

        Returns
        -------
        tuple of ndarray
            Partial fugacity coefficients for vapor and liquid phases [1]
        """
        R = self.R
        T = self.T
        b = self.b
        bmix = self.bmix

        # Calculate compressibility factor, Z
        rho_vap, rho_liq = self.rho(p)
        Z_vap = p / (rho_vap * R * T)
        Z_liq = p / (rho_liq * R * T)

        # Calculate ∂(nα)/∂n_i, where α=a/bRT
        q1 = -0.4347
        q2 = -0.003654
        ln_gam = np.log(self.model.gam())

        alpha = self.a / b / R / T
        alpha_mix = self.amix / bmix / R / T

        term_1 = q2 * (alpha_mix**2 + alpha**2)
        term_2 = ln_gam + np.log(bmix / b) + b / bmix - 1
        term_3 = q1 + 2 * alpha_mix * q2
        alpha_bar = (q1 * alpha + term_1 + term_2) / term_3

        # Calculate ∂(nb)/∂n_i = b_i
        B = bmix * p / R / T
        B_bar = b * p / R / T

        # Calculate partial fugacity coefficients
        ln_phihat_vap = (
            (B_bar / B) * (Z_vap - 1)
            - np.log(Z_vap - B)
            - alpha_bar
            * np.log((Z_vap + (1 + 2**0.5) * B) / (Z_vap + (1 - 2**0.5) * B))
            / 2**1.5
        )
        ln_phihat_liq = (
            (B_bar / B) * (Z_liq - 1)
            - np.log(Z_liq - B)
            - alpha_bar
            * np.log((Z_liq + (1 + 2**0.5) * B) / (Z_liq + (1 - 2**0.5) * B))
            / 2**1.5
        )

        phihat_vap = np.exp(ln_phihat_vap)
        phihat_liq = np.exp(ln_phihat_liq)
        phihat_liq *= self.p(rho_liq) / p
        return phihat_vap, phihat_liq

    # =========================================================================
    # Phase equilibrium calculation
    # =========================================================================

    def cal_vle(
        self, p=None, T=None, x=None, y=None, tol=1e-4, max_iter=500, psat=None, z=None
    ):
        """Calculate vapor-liquid equilibrium (VLE).

        Valid input combinations are (p, x), (p, y), (T, x) and (T, y).

        Parameters
        ----------
        p : float
            Pressure [bar]
        T : float
            Temperature [K]
        x : numpy.ndarray
            Liquid-phase mole fraction [1]
        y : numpy.ndarray
            Vapor-phase mole fraction [1]
        tol : float, optional
            Tolerance for convergence. The default is 1e-4.
        max_iter : int, optional
            Maximum number of iterations. The default is 500.
        psat : numpy.ndarray, optional
            Vapor pressure of each component [bar]. This parameter is for pT
            calculation.
        z : numpy.ndarray, optional
            Feed mole fraction [1]. This parameter is for pT calculation.

        Returns
        -------
        tuple of (float, float, ndarray, ndarray)
            Pressure [bar], temperature [K], liquid-phase mole fraction [1],
            and vapor-phase mole fraction [1]
        """
        # Determine the state based on input combination
        if p is not None and x is not None and T is None and y is None:
            state = 1  # (p, x) provided
        elif p is not None and y is not None and T is None and x is None:
            state = 2  # (p, y) provided
        elif T is not None and x is not None and p is None and y is None:
            state = 3  # (T, x) provided
        elif T is not None and y is not None and p is None and x is None:
            state = 4  # (T, y) provided
        elif p is not None and T is not None and x is None and y is None:
            return self.cal_vle_pT(p, T, tol=tol, max_iter=max_iter, psat=psat, z=z)
        else:
            raise ValueError(
                "Invalid combination of inputs. Allowed combinations are: "
                "(p, x), (p, y), (T, x), (T, y), and (p, T)."
            )

        # Set initial guesses
        if state == 1 or state == 2:  # p provided
            T = 273.15
        if state == 3 or state == 4:  # T provided
            p = 1
        if state == 1 or state == 3:  # x provided
            z1 = x
            z2 = x
        if state == 2 or state == 4:  # y provided
            z1 = y
            z2 = y

        for iteration in range(max_iter):
            # Calculate mole fractions
            if state == 1 or state == 3:  # z1: liquid, z2: vapor
                # Calculate phi_hat for vapor and liquid phases
                self.set_state(T, z1)
                _, phi_liq = self.phi_hat(p)

                self.set_state(T, z2)
                phi_vap, _ = self.phi_hat(p)

                # Calculate vapor composition
                z2 = z1 * phi_liq / phi_vap
                z_sum = z2.sum()
                z2 /= z_sum

            if state == 2 or state == 4:  # z1: vapor, z2: liquid
                # Calculate phi_hat for vapor and liquid phases
                self.set_state(T, z1)
                phi_vap, _ = self.phi_hat(p)

                self.set_state(T, z2)
                _, phi_liq = self.phi_hat(p)

                # Calculate liquid composition
                z2 = z1 * phi_vap / phi_liq
                z_sum = z2.sum()
                z2 /= z_sum

            # Check convergence
            if abs(z_sum - 1.0) < tol:
                if state == 1 or state == 3:
                    y = z2
                if state == 2 or state == 4:
                    x = z2

                return p, T, x, y

            # Update pressure or temperature with sigmoid damping
            if state == 1:
                T /= 0.8 + 0.4 / (1 + np.exp(1 - z_sum))  # 0.8 - 1.2
            if state == 2:
                T *= 0.8 + 0.4 / (1 + np.exp(1 - z_sum))
            if state == 3:
                slope = 1 + 3 / (1 + np.exp(10 - p))  # High p -> fast converge
                p *= 2 / (1 + np.exp(slope * (1 - z_sum)))  # 0.0 - 2.0
            if state == 4:
                slope = 1 + 3 / (1 + np.exp(10 - p))
                p /= 2 / (1 + np.exp(slope * (1 - z_sum)))

            # print(f"Iteration {iteration}: ",
            #       f"p = {p:.4f}, T = {T:.4f}, z1 = {z1}, z2 = {z2}",
            #       f"z_sum = {z_sum:.4f}")

        else:
            print("The VLE calculation did not converged.")
            if state == 1 or state == 3:
                y = z2
            if state == 2 or state == 4:
                x = z2

            return p, T, x, y

    def cal_vle_pT(self, p, T, tol=1e-4, max_iter=500, psat=None, z=None):
        """Calculate vapor-liquid equilibrium by flash calculation.

        Parameters
        ----------
        T : float
            Temperature [K]
        p : float
            Pressure [bar]
        psat : numpy.ndarray
            Vapor pressure of each component [bar]
        tol : float, optional
            Tolerance for convergence. The default is 1e-4.
        max_iter : int, optional
            Maximum number of iterations. The default is 500.
        psat : numpy.ndarray, optional
            Vapor pressure of each component [bar]. This parameter is for pT
            calculation.
        z : numpy.ndarray, optional
            Feed mole fraction [1]. This parameter is for pT calculation.

        Returns
        -------
        tuple of (float, float, ndarray, ndarray)
            Pressure [bar], temperature [K], liquid-phase mole fraction [1],
            and vapor-phase mole fraction [1]
        """

        def rachford_rice(V, z, K):
            F = np.sum(z * (K - 1) / (1 + V * (K - 1)))
            if F > V:
                return V
            if F < V - 1:
                return V - 1
            else:
                return F

        n = len(self.Tc)

        # Calculate vapor pressure of each component
        if psat is None:
            psat = np.array([self.cal_vle(T=T, x=x)[0] for x in np.eye(n)])

        # Calculate x, y assuming Raoult's law
        z = np.ones(n) / n if z is None else z
        V = 0.5
        K = psat / p

        x = z / (V * (K - 1) + 1)
        x /= np.sum(x)

        y = K * x
        y /= np.sum(y)

        # Do flash calculation
        for iteration in range(max_iter):
            V_old = V
            x_old = x
            y_old = y

            # Calculate K
            self.set_state(T, x)
            _, phi_liq = self.phi_hat(p)

            self.set_state(T, y)
            phi_vap, _ = self.phi_hat(p)

            K = phi_liq / phi_vap

            # Renew V
            F = np.sum(z * (K - 1) / (1 + V * (K - 1)))
            dF = -np.sum(z * (K - 1) ** 2 / (1 + V * (K - 1)) ** 2)
            V = V - F / dF

            # Renew x, y
            x = z / (V * (K - 1) + 1)
            x /= x.sum()

            y = K * x
            y /= y.sum()

            # Check for convergence
            V_cond = abs(V - V_old) <= tol
            x_cond = np.any(abs(x - x_old) <= tol)
            y_cond = np.any(abs(y - y_old) <= tol)
            xy_cond = np.any(abs(x - y) <= tol)

            if V_cond or x_cond or y_cond or xy_cond:
                break

        return p, T, x, y


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Define system
#     pr = Peng_Robinson()
#     pr.add_comp("CC")
#     pr.add_comp("CCCCCCC")

#     # Calculate VLE
#     xs = np.linspace(0, 1, 21)

#     pTxy = []
#     for x_ in xs:
#         p, T, x, y = pr.cal_vle(T=400, x=np.array([x_, 1 - x_]))  # BUBL p

#         print(f"p = {p:.4f}, T = {T:.4f}, x = {x[0]:.4f}, y = {y[0]:.4f}")
#         pTxy.append([p, T, x[0], y[0]])
#     pTxy = np.array(pTxy)

#     # Plot VLE
#     plt.plot(pTxy[:, 2], pTxy[:, 0])
#     plt.plot(pTxy[:, 3], pTxy[:, 0])
#     plt.xlabel("x1, y1")
#     plt.ylabel("p (bar)")
#     plt.title("VLE of ethane (1) and heptane (2) at 400 K")
