import json
from os.path import join as opj

import numpy as np
from pathlib import Path
from rdkit import Chem
from scipy.linalg import fractional_matrix_power
from scipy.spatial import distance_matrix


FILE_DIR = Path(__file__).parent


class COSMOSAC:
    # Some codes are from https://doi.org/10.1021/acs.jctc.9b01016

    def __init__(self, version=2010, predict=True):
        # version and system
        self.predict = predict
        if predict:
            self.ver = 2010
        else:
            self.ver = version

        self.x = []  # liquid mole fraction
        self.T = 0  # system temperature

        # global parameters
        if predict:
            with open(opj(FILE_DIR, "model_parameters", "nhb.json"), "rb") as f:
                self.nhb_param = json.load(f)
            with open(opj(FILE_DIR, "model_parameters", "oh.json"), "rb") as f:
                self.oh_param = json.load(f)
            with open(opj(FILE_DIR, "model_parameters", "ot.json"), "rb") as f:
                self.ot_param = json.load(f)
            with open(opj(FILE_DIR, "model_parameters", "vol.json"), "rb") as f:
                self.vol_param = json.load(f)

        self._q0 = 79.53  # area normalization parameter [Å**2]
        self._r0 = 66.69  # volume normalization parameter [Å**3]
        self._z = 10  # coordination number
        self._sighb = 0.0084  # hydrogen bonding screening charge [e/Å**2]
        self._R = 1.987204258e-3  # gas constant [kcal/K/mol]
        self._fdecay = 0.52928 ** (-2)  # unit conversion parameter [1]
        self._sig0 = 0.007  # hydrogen bondable screening charge [e/Å**2]
        self._AES = 6525.69  # electrostatic constant A [kcal*ang**4/mol/e**2]
        # electrostatic constant B [kcal*Å**4*K**2/mol/e**2]
        self._BES = 1.4859e8

        # effective area [Å**2], number of sigma profiles,
        # hydrogen bonding parameter [kcal*Å^4/mol/e^2],
        # electrostatic parameter [kcal*Å^4/mol/e^2]
        self._aeff, self._num_sp, self._chb, self._cES = self._get_var()
        self._reff = np.sqrt(self._aeff / np.pi)  # effective radius, [Å]

        # molecular information from COSMO calculation
        self.A = np.array([])  # cavity area
        self.V = np.array([])  # cavity volume
        # sigma profile * area
        self.psigA = np.array([]).reshape(0, self._num_sp, 51)
        self.ek = np.array([])  # dispersive parameter /K
        self.name = []  # molecule name
        self.dnatr = []  # molecular dispersive type

        # atom radius, [Å]
        self._rc = {
            "H": 0.31,
            "He": 0.28,
            "Li": 1.28,
            "Be": 0.96,
            "B": 0.84,
            "C": 0.76,  # sp3 hybridization, sp2: 0.73 sp1: 0.69
            "N": 0.71,
            "O": 0.66,
            "F": 0.57,
            "Ne": 0.58,
            "Na": 1.66,
            "Mg": 1.41,
            "Al": 1.21,
            "Si": 1.11,
            "P": 1.07,
            "S": 1.05,
            "Cl": 1.02,
            "Ar": 1.06,
            "K": 2.03,
            "Ca": 1.76,
            "Sc": 1.70,
            "Ti": 1.60,
            "V": 1.53,
            "Cr": 1.39,
            "Mn": 1.39,  # l.s.; h.s.: 1.61
            "Fe": 1.32,  # l.s.; h.s.: 1.52
            "Co": 1.26,  # l.s.; h.s.: 1.50
            "Ni": 1.24,
            "Cu": 1.32,
            "Zn": 1.22,
            "Ga": 1.22,
            "Ge": 1.20,
            "As": 1.19,
            "Se": 1.20,
            "Br": 1.20,
            "Kr": 1.16,
            "Rb": 2.20,
            "Sr": 1.95,
            "Y": 1.90,
            "Zr": 1.75,
            "Nb": 1.64,
            "Mo": 1.54,
            "Tc": 1.47,
            "Ru": 1.46,
            "Rh": 1.42,
            "Pd": 1.39,
            "Ag": 1.45,
            "Cd": 1.44,
            "In": 1.42,
            "Sn": 1.39,
            "Sb": 1.39,
            "Te": 1.38,
            "I": 1.39,
            "Xe": 1.40,
            "Cs": 2.44,
            "Ba": 2.15,
            "La": 2.07,
            "Ce": 2.04,
            "Pr": 2.03,
            "Nd": 2.01,
            "Pm": 1.99,
            "Sm": 1.98,
            "Eu": 1.98,
            "Gd": 1.96,
            "Tb": 1.94,
            "Dy": 1.92,
            "Ho": 1.92,
            "Er": 1.89,
            "Tm": 1.90,
            "Yb": 1.87,
            "Lu": 1.87,
            "Hf": 1.75,
            "Ta": 1.70,
            "W": 1.62,
            "Re": 1.51,
            "Os": 1.44,
            "Ir": 1.41,
            "Pt": 1.36,
            "Au": 1.36,
            "Hg": 1.32,
            "Tl": 1.45,
            "Pb": 1.46,
            "Bi": 1.48,
            "Po": 1.40,
            "At": 1.50,
            "Rn": 1.50,
            "Fr": 2.60,
            "Ra": 2.21,
            "Ac": 2.15,
            "Th": 2.06,
            "Pa": 2.00,
            "U": 1.96,
            "Np": 1.90,
            "Pu": 1.87,
            "Am": 1.80,
            "Cm": 1.69,
        }

        self._ang_au = 0.52917721067  # unit change [Å/atomic unit]

    def _get_var(self):
        """Return COSMO-SAC variables according to version.

        Parameters
        ----------
        version : int
            COSMO-SAC version (2002, 2010, 2013, 2019)

        Returns
        -------
        tuple of (aeff, n_psig, chb, cES)
            - aeff: effective surface area [Å²]
            - n_psig: number of sigma profiles
            - chb: hydrogen bonding parameter
            - cES: electrostatic parameter function
        """
        version_params = {
            2002: {
                "aeff": 7.5,
                "n_psig": 1,
                "chb": np.array([[85580]]),
                "cES": lambda T: 8233.36,
            },
            2010: {
                "aeff": 7.25,
                "n_psig": 3,  # nhb, OH, OT
                "chb": np.array(
                    [[0, 0, 0], [0, 4013.78, 3016.43], [0, 3016.43, 932.31]]
                ),
                "cES": lambda T: self._AES + self._BES / T / T,
            },
            2019: {
                "aeff": 7.25,
                "n_psig": 4,  # nhb, OH, OT, COOH
                "chb": np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 4013.78, 3016.43, 3020.18],
                        [0, 3016.43, 932.31, 1872.84],
                        [0, 3020.18, 1872.84, 2225.67],
                    ]
                ),
                "cES": lambda T: self._AES + self._BES / T / T,
            },
        }
        version_params[2013] = version_params[2010]  # 2013 is the same as 2010

        if self.ver not in version_params:
            raise ValueError(f"Version must be one of: {tuple(version_params.keys())}")

        params = version_params[self.ver]
        return params["aeff"], params["n_psig"], params["chb"], params["cES"]

    def _is_from_ms(self, file):
        """Find if the file is from the databases or Material Studio.

        The databases are VT (2006) and UC (2020).

        Parameters
        ----------
        file : str
            The name of the file.

        Returns
        -------
        bool
            True if the file is from Material Studio, False if it is from the
            databases.

        Raises
        ------
        ValueError
            If the file is not interpreted as cosmo file. It is determined by
            the first line of the file.
        """
        # Open file
        opened_file = open(file, "r")
        line = opened_file.readline()

        # Check the origin
        if "COSMO Results from DMol3" in line:  # Material Studio 2017
            return True
        elif "text" in line:  # VT 2006, UD 2020, or KU 2023 database
            return False
        else:
            raise ValueError(f"The file {file} is not interpreted as cosmo file.")

    def _get_cosmo_from_ms(self, file):
        """Get COSMO properties from the cosmo file in Material Studio.

        This code reads the KU database's COSMO files.

        Parameters
        ----------
        opened_file : _io.TextIOWrapper
            Opened file.

        Returns
        -------
        A : float
            Cavity area.
        V : float
            Cavity volume.
        atom : numpy.ndarray of shape=(num_atom,)
            Atom symbols sorted by index in the cosmo file.
        coord : numpy.ndarray of shape=(num_atom, 3)
            The x, y, z coordinates of the atoms.
        seg : numpy.ndarray of shape=(num_seg, 6)
            The list of atom index, x, y, z position, segment area, and charge
            per segment area.
        """
        # Initialize flags and data storage
        flag = "default"
        atom = []  # Atom symbols
        coord = []  # Atom coordinates
        seg = []  # Segment information
        A = None  # Cavity area
        V = None  # Cavity volume
        ang_per_au = self._ang_au

        # Read file and extract data
        with open(file, "r") as f:
            for line in f:
                # Update parsing flag based on section headers
                if "$coordinates xyz [au]" in line:
                    flag = "coordinate"
                    continue
                elif "n  atom        position (X, Y, Z) [au]" in line:
                    flag = "segment"
                    continue
                elif "$end" in line and flag == "coordinate":
                    flag = "default"
                    continue

                # Parse data based on current flag
                if "Surface area of cavity" in line:
                    A = float(line.split()[7])  # [au**2]
                elif "Total Volume of cavity" in line:
                    V = float(line.split()[6])  # [au**3]
                elif flag == "coordinate" and "$end" not in line:
                    parts = line.split()
                    atom.append(parts[0])
                    coord.append([float(x) for x in parts[1:4]])  # [au]
                elif flag == "segment" and line.strip():  # not empty line
                    parts = line.split()
                    seg.append(
                        [int(parts[1]) - 1]
                        + [float(x) for x in parts[2:5] + parts[6:8]]
                    )

        # Convert lists to numpy arrays
        atom = np.array(atom)
        coord = np.array(coord)
        seg = np.array(seg)

        # Convert units from atomic units to angstroms
        A *= ang_per_au**2  # [Å**2]
        V *= ang_per_au**3  # [Å**3]
        coord *= ang_per_au  # [Å]
        seg[:, 1:4] *= ang_per_au  # [Å]
        seg[:, 4] *= ang_per_au**2  # [Å**2]
        seg[:, 5] /= ang_per_au**2  # [e/Å**2]

        return A, V, atom, coord, seg

    def _get_cosmo_from_not_ms(self, file):
        """Get COSMO properties from the cosmo file not in Material Studio.

        This code reads the VT and UD databases' COSMO files.

        Parameters
        ----------
        opened_file : _io.TextIOWrapper
            Opened file.

        Returns
        -------
        A : float
            Cavity area.
        V : float
            Cavity volume.
        atom : numpy.ndarray of shape=(num_atom,)
            Atom symbols sorted by index in the cosmo file.
        coord : numpy.ndarray of shape=(num_atom, 3)
            The x, y, z coordinates of the atoms.
        seg : numpy.ndarray of shape=(num_seg, 6)
            The list of atom index, x, y, z position, segment area, and charge
            per segment area.
        """
        # Initialize storage
        atom = []  # Atom symbols
        coord = []  # Atom coordinates
        seg = []  # Segment information
        A = None  # Cavity area
        V = None  # Cavity volume
        flag = "default"

        # Read and parse file
        with open(file, "r") as f:
            for line in f:
                # Update parsing flag based on section headers
                if "!DATE" in line:
                    flag = "coordinate"
                    continue
                elif "n   atom        position (X, Y, Z) [au]" in line:
                    flag = "segment"
                    continue
                elif "end" in line and flag == "coordinate":
                    flag = "default"
                    continue

                # Parse data based on current flag
                if "Total surface area of cavity" in line:
                    A = float(line.split()[7])  # [Å**2]
                elif "Total volume of cavity" in line:
                    V = float(line.split()[6])  # [Å**3]
                elif flag == "coordinate" and "end" not in line:
                    parts = line.split()
                    atom.append(parts[7])
                    coord.append([float(x) for x in parts[1:4]])  # [Å]
                elif flag == "segment" and line.strip():  # not empty line
                    parts = line.split()
                    seg.append(
                        [int(parts[1]) - 1]
                        + [float(x) for x in parts[2:5] + parts[6:8]]
                    )
                    # [0], [au], [au], [au], [Å**2], [e/Å**2]

        # Convert to numpy arrays
        atom = np.array(atom)
        coord = np.array(coord)
        seg = np.array(seg)

        # Convert units from atomic units to angstroms
        seg[:, 1:4] *= self._ang_au  # [Å]

        return A, V, atom, coord, seg

    def get_cosmo(self, file):
        """Get COSMO properties from the cosmo extension file.

        Parameters
        ----------
        file : str
            The name of the cosmo file.

        See Also
        --------
        is_from_ms
            Function to check if the file is from databases or Material Studio.
        get_cosmo_from_ms, get_cosmo_from_not_ms
            Functions to get COSMO informations.
        """
        if self._is_from_ms(file):
            return self._get_cosmo_from_ms(file)
        else:
            return self._get_cosmo_from_not_ms(file)

    def get_bond(self, atom, coord):
        """Get bond matrix.

        Parameters
        ----------
        atom : numpy.ndarray of shape=(num_atom,)
            Atom symbols sorted by index in the cosmo file.
        coord : numpy.ndarray of shape=(num_atom, 3)
            The x, y, z coordinates of the atoms.

        Returns
        -------
        bond : numpy.ndarray of shape=(num_atom, num_atom)
            The bond matrix. If two atoms are bonded, their entry is 1, else 0.
        """
        d_atom = distance_matrix(coord, coord)  # Distance between atoms
        rc = np.array([self._rc[a] for a in atom])  # Radii of atoms

        mask = d_atom < 1.15 * (rc[:, np.newaxis] + rc[np.newaxis, :])
        bond = np.where(mask, 1, 0)
        np.fill_diagonal(bond, 0)  # Atoms do not bond with themselves.

        return bond

    def get_atom_type(self, atom, bond):
        """Get hybridization and sigma profile types for each atom.

        The dispersive natures are as below.
        DSP_WATER : WATER in this code. This indicates water.
        DSP_COOH : COOH in this code. This indicates a molecule with a carboxyl
        group.
        DSP_HB_ONLY_ACCEPTOR : HBOA in this code. The molecule contains any of
        the atoms O,N, or F but no H atoms bonded to any of these O, N, or F.
        DSP_HB_DONOR_ACCEPTOR : HBDA in this code. The molecule contains any of
        the functional groups NH, OH, or FH (but not OH of COOH or water).
        DSP_NHB : NHB in this code. This indicates that the molecule is non-
        hydrogen-bonding.

        The dispersion types are as below.
        C(sp3) : C bonded to 4 others.
        C(sp2) : C bonded to 3 others.
        C(sp) : C bonded to 2 others.
        N(sp3) : N bonded to three others.
        N(sp2) : N bonded to two others.
        N(sp) : N bonded to one other.
        -O- : O(sp3) in this code. O bonded to 2 others.
        =O : O(sp2) in this code. Double-bonded O.
        F : F bonded to one other.
        Cl : Cl bonded to one other.
        H(water) : H in water.
        H(OH) : H-O bond but not water.
        H(NH) : H bonded to N.
        H(other) : H otherwise.
        other : Undifined.

        The hydrogen-bonding types are as below.
        OH : if the atom is O and is bonded to an H, or vice versa.
        OT : if the atom is O and is bonded to an atom other than H, or if the
        atom is H and is bonded to N or F.
        COOH : if the atoms are C, O, H and are in the carboxyl group.
        NHB : otherwise.

        Parameters
        ----------
        version : {2002, 2010, 2013, 2019}
            The COSMO-SAC version.
        atom : numpy.ndarray of shape=(num_atom,)
            Atom symbols sorted by index in the cosmo file.
        bond : numpy.ndarray of shape=(num_atom, num_atom)
            The bond matrix. If two atoms are bonded, their entry is 1, else 0.

        Returns
        -------
        dtype : list of shape=(num_atom,)
            The dispersion type for each atom.
        stype : list of shape=(num_atom,)
            The hydrogen-bonding type for each atom.
        dnatr : {"NHB", "HBOA", "HBDA", "WATER", "COOH"}
            The dispersive nature of the molecule.
        """
        dtype = ["other"] * len(atom)  # hybridization type
        stype = ["NHB"] * len(atom)  # sigma profile type
        dnatr = "NHB"  # dispersive nature of molecule
        dntype = set()  # dispersive nature type of atoms

        # no types for COSMO-SAC 2002
        if self.ver == 2002:
            return dtype, stype, dnatr

        # {atom type: {bonded atoms: (dtype, stype, dnatr), ...}, ...}
        # This assumes that all atoms are belong to NHB, OT and H(other).
        atom_prop = {
            "C": {
                2: ("C(sp)", "NHB", "NHB"),
                3: ("C(sp2)", "NHB", "NHB"),
                4: ("C(sp3)", "NHB", "NHB"),
            },
            "O": {
                1: ("O(sp2)", "OT", "HBOA"),
                2: ("O(sp3)", "OT", "HBOA"),
            },
            "N": {
                1: ("N(sp)", "OT", "HBOA"),
                2: ("N(sp2)", "OT", "HBOA"),
                3: ("N(sp3)", "OT", "HBOA"),
            },
            "F": {1: ("F", "OT", "HBOA")},
            "Cl": {1: ("Cl", "NHB", "NHB")},
            "H": {1: ("H(other)", "NHB", "NHB")},
        }

        for i, atom_type in enumerate(atom):
            # Get dictionary of index and atom types bonded with atom i
            ard_i = {j: atom[j] for j in np.flatnonzero(bond[i])}

            # If the atom is in the difined properties
            if atom_type in atom_prop:
                # Get atom types, else get ("Undifined", 0)
                dtype[i], stype[i], dntype_i = atom_prop[atom_type].get(
                    len(ard_i), ("other", "NHB", "NHB")
                )
                dntype.add(dntype_i)

            # Find H near N, and renew the types of H
            if atom_type == "H" and "N" in ard_i.values():
                dtype[i] = "H(NH)"
                stype[i] = "OT"
                dntype.add("HBDA")

            # Find H in HF, and renew the types of H
            if atom_type == "H" and "F" in ard_i.values():
                stype[i] = "OT"
                dntype.add("HBDA")

            # Find atom type for -OH, H2O, and COOH
            if atom_type == "H" and "O" in ard_i.values():
                # # Renew the typs of H and O in OH
                # Renew the types of H
                dtype[i] = "H(OH)"
                stype[i] = "OH"

                # Find the atom index of O in OH
                j = list(ard_i.keys())[0]
                ard_j = {k: atom[k] for k in np.flatnonzero(bond[j])}
                # Renew the types of O in -OH
                stype[j] = "OH"
                dntype.add("HBDA")

                # # Further find H-OH and CO-OH
                # if the O in -OH has not two bonds, stop searching
                if len(ard_j) != 2:
                    break

                # Find atom index of neighber of O in -OH, but not H in -OH
                k = [k for k in ard_j.keys() if k != i][0]
                ard_k = {m: atom[m] for m in np.flatnonzero(bond[k])}

                # if atom k is H, that is, if the molecule is water, renew the
                # dtype of the Hs in H2O and stop searching
                if atom[k] == "H":
                    dtype[i] = "H(water)"
                    dtype[k] = "H(water)"
                    dntype.add("WATER")
                    break

                # # Further find COOH
                # if the atom k is not the C in part of COOH, stop searching
                if not (
                    atom[k] == "C"
                    and len(ard_k) == 3
                    and list(ard_k.values()).count("O") == 2
                ):
                    break

                # Find the O, neighber of C in -COH, but not in O in -COH
                m = [m for m in ard_k.keys() if (m != j and ard_k[m] == "O")][0]
                ard_m = {n: atom[n] for n in np.flatnonzero(bond[m])}

                # if the atom m is -O-, not =O, stop searching
                if len(ard_m) != 1:
                    break

                # Renew i(H), j(O), k(C) and m(O) as the part of COOH
                dntype.add("COOH")
                if self.ver == 2019:
                    stype[i] = "COOH"
                    stype[j] = "COOH"
                    stype[m] = "COOH"

        # find the dispersive nature of the molecule
        if "HBOA" in dntype:
            dnatr = "HBOA"
        if "HBDA" in dntype:
            dnatr = "HBDA"
        if "WATER" in dntype:
            dnatr = "WATER"
        if "COOH" in dntype:
            dnatr = "COOH"

        return dtype, stype, dnatr

    def get_sigma(self, atom, seg, stype):
        """Get sigma profiles.

        Parameters
        ----------
        atom : numpy.ndarray of shape=(num_atom,)
            Atom symbols sorted by index in the cosmo file.
        seg : numpy.ndarray of shape=(num_seg, 6)
            The list of atom index, x, y, z position, segment area, and charge
            per segment area.
        stype : list of shape=(num_atom,)
            The sigma profile type for each atom.

        Returns
        -------
        psigA : numpy.ndarray of shape=(num_sp, 51)
            The sigma profiles of the molecule. The number of sigma profiles is
            dependent on the version.
            {version: num_sp} = {2002: 1, 2010: 3, 2013: 3, 2019: 4}
        """
        # import global parameters
        reff = self._reff
        num_sp = self._num_sp

        # Set sigma profile types to integers
        type_mapping = {"NHB": 0, "OH": 1, "OT": 2, "COOH": 3}
        stype_int = np.array([type_mapping[element] for element in stype])

        # Define segment informations
        seg_atom_index = np.int32(seg[:, 0])
        seg_atom = atom[seg_atom_index]
        seg_stype = stype_int[seg_atom_index]
        seg_coord = seg[:, 1:4]
        seg_area = seg[:, 4]
        seg_charge = seg[:, 5]

        # Calculate radii of the segments and distances between the segments
        r = np.sqrt(seg_area / np.pi)
        d = distance_matrix(seg_coord, seg_coord)

        # Calculate averaged surface charges of the segments
        rcal = r**2 * reff**2 / (r**2 + reff**2)
        dcal = np.exp(-self._fdecay * d**2 / (r**2 + reff**2).reshape(-1, 1))

        upper = np.einsum("n,n,mn->m", seg_charge, rcal, dcal)
        lower = np.einsum("n,mn->m", rcal, dcal)

        seg_avg_charge = upper / lower

        # Decide sigma profile types
        # Initialize all segments as NHB (type 0)
        sig_type = np.int32(np.zeros(len(seg)))

        # OH sigma profile (type 1) conditions
        oh_oxygen_mask = (seg_atom == "O") & (seg_stype == 1) & (seg_avg_charge > 0)
        oh_hydrogen_mask = (seg_atom == "H") & (seg_stype == 1) & (seg_avg_charge < 0)

        # OT sigma profile (type 2) conditions
        ot_acceptor_mask = (
            ((seg_atom == "O") | (seg_atom == "N") | (seg_atom == "F"))
            & (seg_stype == 2)
            & (seg_avg_charge > 0)
        )
        ot_hydrogen_mask = (seg_atom == "H") & (seg_stype == 2) & (seg_avg_charge < 0)

        # Update sigma types
        sig_type = np.where(oh_oxygen_mask | oh_hydrogen_mask, 1, sig_type)
        sig_type = np.where(ot_acceptor_mask | ot_hydrogen_mask, 2, sig_type)

        # Find COOH sigma profile
        sig_type = np.where(seg_stype == 3, 3, sig_type)

        # Calculate sigma profiles
        sig = np.linspace(-0.025, 0.025, 51)

        left = np.int32(np.floor((seg_avg_charge - sig[0]) / 0.001))
        w = (sig[left + 1] - seg_avg_charge) / 0.001

        psigA = np.zeros((num_sp, 51))
        np.add.at(psigA, (sig_type, left), w * seg_area)
        np.add.at(psigA, (sig_type, left + 1), (1 - w) * seg_area)

        if self.ver != 2002:
            phb = 1 - np.exp(-(sig**2) / 2 / self._sig0**2)
            psigA[0] = psigA[0] + np.sum(psigA[1:], axis=0) * (1 - phb)
            psigA[1:] = psigA[1:] * phb

        return psigA

    def get_dsp(self, dtype):
        """Get the dispersive nature of the molecule.

        Parameters
        ----------
        dtype : list of shape=(num_atom,)
            The dispersion type for each atom.

        Returns
        -------
        ek : float
            Dispersive parameter.
        """
        if self.ver == 2002 or self.ver == 2010 or "other" in dtype:
            ek = None
            return ek

        # dispersive parameters
        ddict = {
            "C(sp3)": 115.7023,
            "C(sp2)": 117.4650,
            "C(sp)": 66.0691,
            "N(sp3)": 15.4901,
            "N(sp2)": 84.6268,
            "N(sp)": 109.6621,
            "O(sp3)": 95.6184,  # -O-
            "O(sp2)": -11.0549,  # =O
            "F": 52.9318,
            "Cl": 104.2534,
            "H(water)": 58.3301,
            "H(OH)": 19.3477,
            "H(NH)": 141.1709,
            "H(other)": 0,
        }

        # calculate the dispersive parameter of the molecule
        ek = np.vectorize(ddict.get)(dtype)
        ek = np.sum(ek) / np.count_nonzero(ek)

        return ek

    def add_comp(self, file=None, SMILES=None, name=None):
        """Add a component to the COSMO-SAC using the COSMO file of ML model.

        Parameters
        ----------
        file : str, optional
            The path of the cosmo file. The default is None.
        smiles : str, optional
            SMILES string of the compound. The default is None.
        name : str, optional
            The name of the component. The default is None.

        Raises
        ------
        ValueError : If the COSMO file nor SMILES string is not provided.
        """
        if file is not None:  # If COSMO file is given
            self.add_comp_from_COSMO_file(file, name=name)

        elif SMILES is not None:  # If SMILES string is given
            file_dir = get_COSMO_file_dir(SMILES)

            if file_dir is not None:  # If the COSMO file exists
                self.add_comp_from_COSMO_file(file_dir)
            else:  # Else, predict the COSMO properties
                self.add_comp_from_GCGCN_CPU(SMILES)

        else:
            raise ValueError("The COSMO file or SMILES string is necessary.")

    def add_comp_from_COSMO_file(self, file, name=None):
        """Add a component to the COSMO-SAC object.

        Parameters
        ----------
        file : str
            The path to the cosmo file.
        name : str, optional
            The name of the component.
        """
        # get cosmo parameters
        A, V, atom, coord, seg = self.get_cosmo(file)
        bond = self.get_bond(atom, coord)
        dtype, stype, dnatr = self.get_atom_type(atom, bond)
        psigA = self.get_sigma(atom, seg, stype).reshape(1, self._num_sp, 51)
        ek = self.get_dsp(dtype)

        # add component
        self.A = np.append(self.A, A)
        self.V = np.append(self.V, V)
        self.psigA = np.vstack((self.psigA, psigA))
        self.ek = np.append(self.ek, ek)
        self.dnatr.append(dnatr)
        self.name.append(name)

    def add_comp_from_GCGCN_CPU(self, SMILES, name=None):
        """
        Add a component with GC-GCN predictions to COSMO-SAC.

        Parameters
        ----------
        smiles : str
            SMILES string of the component.
        name : str, optional
            Name of the component.
        """
        # Get matrix input
        nfm, efm = get_cosmo_input(SMILES)

        # Predict COSMO properties
        volume = 562 * pred_COSMO_with_numpy(efm, nfm, self.vol_param)[0]

        sigma_profiles = np.zeros((3, 51))
        sigma_profiles[0] = 145 * pred_COSMO_with_numpy(efm, nfm, self.nhb_param)
        sigma_profiles[1] = 7 * pred_COSMO_with_numpy(efm, nfm, self.oh_param)
        sigma_profiles[2] = 16 * pred_COSMO_with_numpy(efm, nfm, self.ot_param)
        sigma_profiles = np.where(sigma_profiles < 0, 0, sigma_profiles)
        sigma_profiles = sigma_profiles.reshape(1, self._num_sp, 51)

        # area = np.sum(145*pred_COSMO_with_numpy(efm, nfm, sig_param))
        area = np.sum(sigma_profiles)

        # add component
        self.A = np.append(self.A, area)
        self.V = np.append(self.V, volume)
        self.psigA = np.vstack((self.psigA, sigma_profiles))
        self.name.append(name)

    def del_comp(self):
        """Delete a component from the COSMO-SAC object."""
        self.A = np.array([])
        self.V = np.array([])
        self.psigA = np.array([])
        self.ek = np.array([])

        self.dnatr = []
        self.name = []

    def cal_DW(self, T):
        """Calculate the exchange energy.

        The exchange energy has the values for each charge density combinations
        and sigma profile type combinations, therefore having the shape of
        (num_sp, num_sp, 51, 51).

        Parameters
        ----------
        T : float
            The system temperature.

        Returns
        -------
        DW : numpy.ndarray of shape=(num_sp, num_sp, 51, 51)
            The exchange energy.
        """
        # Initialize parameters
        sig = np.linspace(-0.025, 0.025, 51)
        sigT = sig.reshape(-1, 1)
        DW = np.zeros((self._num_sp, self._num_sp, 51, 51))

        # Calculate exchange energy for each pair of sigma profile types
        for i in range(self._num_sp):
            for j in range(i + 1):
                # Calculate hydrogen bonding contribution
                if self.ver == 2002:
                    acchb = np.maximum(sig, sigT) - self._sighb
                    donhb = np.minimum(sig, sigT) + self._sighb
                    maxacc = np.where(acchb > 0, acchb, 0)
                    mindon = np.where(donhb < 0, donhb, 0)
                    chb_part = -self._chb[i][j] * maxacc * mindon
                else:
                    mask = (sig * sigT) < 0
                    chb_part = np.where(mask, self._chb[i, j] * (sig - sigT) ** 2, 0)

                # Calculate total exchange energy
                DW[i, j] = DW[j, i] = self._cES(T) * (sig + sigT) ** 2 - chb_part

        return DW

    # =========================================================================
    # Calculate activity coefficients
    # =========================================================================

    def ln_gam_comb(self):
        """Calculate log of combinatory activity coefficients.

        Parameters
        ----------
        None.

        Returns
        -------
        ln_gam_comb : numpy.ndarray of shape=(num_comp,)
            Combinatory activity coefficients of components.
        """
        # calculate normalized areas and volumes
        q = self.A / self._q0
        r = self.V / self._r0
        L = (self._z / 2) * (r - q) - (r - 1)

        theta = q / np.sum(self.x * q)
        phi = r / np.sum(self.x * r)

        # calcualte combinatory activity coefficients
        ln_gam_comb = (
            np.log(phi)
            + self._z * q * np.log(theta / phi) / 2
            + L
            - phi * np.sum(self.x * L)
        )
        return ln_gam_comb

    def cal_psig_mix(self):
        """Calculate the mixture sigma profile of the mixture.

        Parameters
        ----------
        None.

        Returns
        -------
        psig_mix : numpy.ndarray of shape=(num_sp, 51)
            The mixture sigma profiles.
        """
        psig_mix = np.einsum("i,itm->tm", self.x, self.psigA) / np.sum(self.x * self.A)
        return psig_mix

    def ln_gam_res(self):
        """Calculate residual activity coefficients.

        Parameters
        ----------
        None.

        Returns
        -------
        ln_gam_res : numpy.ndarray of shape=(num_comp,)
            Residual activity coefficients of components.
        """
        # calculate intermediate terms
        psig = np.einsum("itm,i->itm", self.psigA, 1 / self.A)
        psig_mix = self.cal_psig_mix()

        exp_DW = np.exp(-self.cal_DW(self.T) / self._R / self.T)

        A_plus = np.einsum("stmn,isn->istmn", exp_DW, psig)  # A^(+)
        A_plus_mix = np.einsum("stmn,sn->stmn", exp_DW, psig_mix)  # A^(+)_mix

        # calculate the segment activity coefficients
        Gam = np.ones(np.shape(psig))
        Gam_mix = np.ones(np.shape(psig_mix))
        diff = 1

        for _ in range(500):
            Gam_old = Gam
            Gam_mix_old = Gam_mix

            # update segment activities
            Gam = 1 / np.einsum("istmn,isn->itm", A_plus, Gam)
            Gam_mix = 1 / np.einsum("stmn,sn->tm", A_plus_mix, Gam_mix)

            # apply damping
            Gam = (1.618 * Gam + Gam_old) / 2.618
            Gam_mix = (1.618 * Gam_mix + Gam_mix_old) / 2.618

            # check convergence
            diff = np.max(
                [
                    np.max(np.abs((Gam - Gam_old) / Gam_old)),
                    np.max(np.abs((Gam_mix - Gam_mix_old) / Gam_mix_old)),
                ]
            )

            if diff <= 1e-4:
                break

        else:
            print("The convergence failed.")

        # calculate residual activity coefficients
        Gam_part = np.log(Gam_mix) - np.log(Gam)
        ln_gam_res = np.einsum("itm,itm->i", self.psigA, Gam_part) / self._aeff

        return ln_gam_res

    def ln_gam_dsp(self):
        """Calculate dispersive activity coefficients.

        Parameters
        ----------
        None.

        Returns
        -------
        ln_gam_dsp : numpy.ndarray of shape=(num_comp,)
            Dispersive activity coefficients of components.
        """
        num_mol = len(self.x)
        ekT = self.ek.reshape(-1, 1)

        # check if dispersion activity coefficients are applicable
        if None in self.ek or None in self.dnatr:
            ln_gam_dsp = [None] * num_mol
            return ln_gam_dsp

        # calculate interaction parameters
        w = np.ones((num_mol, num_mol)) * 0.27027
        wpair = [
            {"WATER", "HBOA"},
            {"COOH", "NHB"},
            {"COOH", "HBDA"},
            {"WATER", "COOH"},
        ]
        for i in range(num_mol):
            for j in range(i):
                if {self.dnatr[i], self.dnatr[j]} in wpair:
                    w[i][j] = w[j][i] = -0.27027

        A = w * (0.5 * (self.ek + ekT) - np.sqrt(self.ek * ekT))  # not area

        # calculate dispersive activity coefficients
        ln_gam_dsp = np.zeros(num_mol)
        for i in range(num_mol):
            for j in range(num_mol):
                if i != j:
                    ln_gam_dsp[i] = ln_gam_dsp[i] + self.x[j] * A[i, j]
                if j > i:
                    ln_gam_dsp[i] = ln_gam_dsp[i] - self.x[i] * self.x[j] * A[i, j]

        return ln_gam_dsp

    def gam(self):
        """Calculate COSMO-SAC activity coefficients.

        Parameters
        ----------
        None.

        Returns
        -------
        gam : numpy.ndarray of shape=(num_comp,)
            Activity coefficients of components.
        """
        # calculate log activity cofficients for each contribution
        ln_gam_comb = self.ln_gam_comb()
        ln_gam_res = self.ln_gam_res()
        ln_gam_dsp = self.ln_gam_dsp()

        # check if dispersion activity coefficients are applicable
        if None in ln_gam_dsp:
            ln_gam = ln_gam_comb + ln_gam_res
        else:
            ln_gam = ln_gam_comb + ln_gam_res + ln_gam_dsp

        # calculate activity coefficients
        gam = np.exp(ln_gam)

        return gam


def get_COSMO_file_dir(SMILES: str) -> (str | None):
    """Get COSMO file's directory if there is already calculated molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of the compound.

    Return
    ------
    str or None : The COSMO file directory if available, else None.

    Raises
    ------
    ValueError : If the molecule is not readable by rdkit.
    """
    with open(opj(FILE_DIR, "InChIKey_to_index.json")) as json_file:
        InChIKey_to_index = json.load(json_file)

    try:
       InChIKey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(SMILES))
    except ValueError:
        raise ValueError("The molecule is not supported.")

    if InChIKey in InChIKey_to_index:  # If InchIKey is in the dctionary keys
        COSMO_file_dir = opj(
            FILE_DIR,
            "cosmo_files",
            f"{InChIKey_to_index[InChIKey]}.cosmo",
        )
        return COSMO_file_dir
    else:
        return None


smarts_cosmo_list = [
    "[CX4H3]",
    "[CX3H2v4]",
    "[CX2H1v4]",
    "[!R;CX4H2]",
    "[!R;CX4H]",
    "[!R;CX4H0]",
    "[!R;CX3H1v4]",
    "[!R;CX3H0v4]",
    "[!R;CX2H0;$([CX2H0](=*)(=*))]",
    "[!R;CX2H0;$([CX2H0](#*))]",
    "[R;CX4H2]",
    "[R;CX4H]",
    "[R;CX4H0]",
    "[R;CX3H1v4]",
    "[R;CX3H0v4]",
    "[R;CX2H0;$([CX2H0](=*)(=*))]",
    "[R;CX2H0;$([CX2H0](#*))]",
    "[R;cX4h2]",
    "[R;cX4h]",
    "[R;cX3h1v4]",
    "[R;cX3h0v4]",
    "[FX1H0]",
    "[ClX1H0]",
    "[BrX1H0]",
    "[IX1H0]",
    "[OX2H1]",
    "[!R;OX2H0]",
    "[R;OX2H0]",
    "[R;oX2h0]",
    "[OX1H0v2]",
    "[NX3H2v3]",
    "[NX3H1v3;!R]",
    "[NX3H1v3;R]",
    "[nX3h1v3;R]",
    "[NX3H0v3;!R]",
    "[NX3H0v3;R]",
    "[nX3h0v3;R]",
    "[NX2H0v3;!R]",
    "[NX2H0v3;R]",
    "[nX2h0v3;R]",
    "[NX1H0v3]",
    "[SX2H1v2]",
    "[SX2H0v2;!R]",
    "[SX2H0v2;R]",
    "[sX2h0v2;R]",
    "[SX1H0v2]",
    "[OX1H0v1]",
    "[NX3H0v4]",
]


def get_cosmo_input(smiles):
    """Generate node and edge feature matrices for GC-GCN.

    Parameters
    ----------
    smiles : str
        SMILES string of the compound.

    Returns
    -------
    nfm, efm : tuple of numpy.ndarray
        Node feature matrix (nfm) of shape (25, len(smarts_list)) and edge
        feature matrix (efm) of shape (25, 25).
    """

    mol = Chem.MolFromSmiles(smiles)

    # Get node feature matrix
    nfm = np.zeros((25, len(smarts_cosmo_list)))

    for smarts_index, smarts in enumerate(smarts_cosmo_list):
        pat = Chem.MolFromSmarts(smarts)
        nfm[mol.GetSubstructMatches(pat), smarts_index] = 1

    # Get edge feature matrix
    efm = Chem.GetAdjacencyMatrix(mol).astype("float64")
    np.fill_diagonal(efm, 1)

    diag = np.diag(np.sum(efm, axis=1))
    diag_half = fractional_matrix_power(diag, -0.5)
    efm = np.matmul(np.matmul(diag_half, efm), diag_half)

    # Padding edge feature matrix
    n_heavyatom = len(efm)
    pad = 25 - n_heavyatom
    efm = np.pad(efm, ((0, pad), (0, pad)), "constant", constant_values=0.0)

    return nfm, efm


def pred_COSMO_with_numpy(efm, nfm, param_list):
    """Predict output using GC-GCN model with learned parameters.

    Parameters
    ----------
    efm : np.ndarray
        Edge feature matrix of shape (25, 25).
    nfm : np.ndarray
        Node feature matrix of shape (25, len(list_smarts)).
    param_list : list of np.ndarray
        List of parameters for the GC-GCN model layers.

    Returns
    -------
    x : numpy.ndarray
        Predicted values as a flat array.
    """
    x = np.dot(efm, nfm)
    x = np.dot(x, param_list[0]) + param_list[1]  # Graph convolution layer
    x = np.where(x > 0, x, 0)  # ReLU

    x = np.dot(efm, x)
    x = np.dot(x, param_list[2]) + param_list[3]
    x = np.where(x > 0, x, 0)

    x = x.reshape(-1)
    x = np.dot(x, np.tile(np.eye(256), (25, 1)))  # Node-wise summation

    x = np.dot(x, param_list[4]) + param_list[5]  # Dense layer
    x = np.where(x > 0, x, 0)

    x = np.dot(x, param_list[6]) + param_list[7]
    x = np.where(x > 0, x, 0)

    x = np.dot(x, param_list[8]) + param_list[9]
    x = np.where(x > 0, x, 0)

    x = np.dot(x, param_list[10]) + param_list[11]
    return x
