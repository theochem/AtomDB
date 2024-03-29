# This file is part of AtomDB.
#
# AtomDB is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# AtomDB is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with AtomDB. If not, see <http://www.gnu.org/licenses/>.

DATASET_PROPERTIES = {
    # General Properties
    # ==================
    "elem": {"TYPE": "GENERAL", "DESC": "Refers to an element from the Periodic Table."},
    "dataset": {
        "TYPE": "GENERAL",
        "DESC": "Aggregated information about molecular structures and properties.",
    },
    "basis": {
        "TYPE": "GENERAL",
        "DESC": "Standard used to approximate the molecular orbitals within a computational method. "
        "Provides a basis for expressing the wavefunction of a molecule. Significantly influence "
        "the accuracy of quantum chemical calculations.",
    },
    # Scalar Properties
    # =================
    "mass": {"TYPE": "SCALAR", "DESC": "Atomic mass of the element."},
    "energy": {
        "TYPE": "SCALAR",
        "DESC": "Energy associated with electronic config of species or energy levels of molecular orbitals.",
    },
    "ip": {
        "TYPE": "SCALAR",
        "DESC": "Ionization Potential (energy req. to remove an electron from atom or species / molecule. "
        "Key parameter in understanding reactivity and electronic structure of species / molecules.",
    },
    "nexc": {
        "TYPE": "SCALAR",
        "DESC": "Electronic State Number. Currently defaults to ground state and is set to 0.",
    },
    "natom": {
        "TYPE": "SCALAR",
        "DESC": "Count of individual atoms within species / molecular system. Fundamental parameter used to "
        "characterize size and complexity of molecules.",
    },
    "nelec": {
        "TYPE": "SCALAR",
        "DESC": "Count of electrons within a species / molecular system. Distribution of electrons among "
        "atomic orbitals determines many chemical and physical properties of species / molecules.",
    },
    "nspin": {
        "TYPE": "SCALAR",
        "DESC": "Count of distinct spin configs available to a species / molecular system. Particularly "
        "relevant in systems with unpaired electrons, where spin multiplicity influences electronic "
        "properties and reactivity.",
    },
    "vdw_radii": {
        "TYPE": "SCALAR",
        "DESC": "Van der Waals radii. Represents effective size of atoms or molecules in terms "
        "of their non-bonded interactions. Often used to estimate intermolecular distances "
        "and important for understanding molecular packing and interactions.",
    },
    "cov_radii": {
        "TYPE": "SCALAR",
        "DESC": "Covalent radii. Represents sizes of atoms based on the lengths of their covalent "
        "bonds in molecules. Important for estimating bond lengths and molecular geometries.",
    },
    "mu": {
        "TYPE": "SCALAR",
        "DESC": "Chemical potential. Partial derivative of the total energy of a system "
        "with respect to the number of electrons. Signifies the energy required to add "
        "an extra electron to a system.",
    },
    "eta": {
        "TYPE": "SCALAR",
        "DESC": "Damping / Broadening factor. Used in computational chemistry calculations. "
        "Conceptual Density Functional Theory (DFT) related property.",
    },
    "rs": {
        "TYPE": "SCALAR",
        "DESC": "Radial Grid. Represents the distances between adjacent grid points on the "
        "grid used to discretize the space around the molecular system. Crucial for "
        "accurate representation of the distribution of electrons and other properties "
        "within the system. Determines resolution and accuracy of calculations "
        "involving the molecular structure.",
    },
    "dens_tot": {
        "TYPE": "SCALAR",
        "DESC": "Total electron density in the system. Includes contributions from all "
        "electrons, regardless of their spin (both and up and down spins)",
    },
    "ked_tot": {
        "TYPE": "SCALAR",
        "DESC": "Total total kinetic energy density in the system. Includes contributions "
        "from all electrons, regardless of their spin (both and up and down spins)",
    },
    # Vector Properties
    # =================
    "mo_e_up": {
        "TYPE": "VECTOR",
        "DESC": "Energy levels of the molecular orbitals for spin-up electrons",
    },
    "mo_e_dn": {
        "TYPE": "VECTOR",
        "DESC": "Energy levels of the molecular orbitals for spin-down electrons",
    },
    "occs_up": {
        "TYPE": "VECTOR",
        "DESC": "Occupation numbers (spin-up). Number of electrons in each molecular orbital for "
        "spin-up electrons",
    },
    "occs_dn": {
        "TYPE": "VECTOR",
        "DESC": "Occupation numbers (spin-down). Number of electrons in each molecular orbital for "
        "spin-down electrons",
    },
    "_orb_dens_up": {
        "TYPE": "VECTOR",
        "DESC": "Orbital densities for spin-up electrons. Describes the probability density "
        "of finding an electron in a particular orbital",
    },
    "_orb_dens_dn": {
        "TYPE": "VECTOR",
        "DESC": "Orbital densities for spin-down electrons. Describes the probability density "
        "of finding an electron in a particular orbital",
    },
    "_orb_ked_up": {
        "TYPE": "VECTOR",
        "DESC": "Kinetic Energy Densities (KED) for spin-up electrons. Describe the "
        "distribution of kinetic energy per unit volume associated with the motion of electrons.",
    },
    "_orb_ked_dn": {
        "TYPE": "VECTOR",
        "DESC": "Kinetic Energy Densities (KED) for spin-down electrons. Describe the "
        "distribution of kinetic energy per unit volume associated with the motion of electrons.",
    },
    # Default Property
    # ================
    "_": {"TYPE": "NA", "DESC": "Property Information Not Found"},
}


def generate_property_docs(properties, n=1, doctype="DOCSTRING"):
    lookup, docs = {"GENERAL": [], "SCALAR": [], "VECTOR": [], "MISCELLANEOUS": []}, "\n"

    for prop in properties:
        attr = DATASET_PROPERTIES.get(prop, DATASET_PROPERTIES["_"])
        if attr["TYPE"] == "GENERAL":
            lookup["GENERAL"].append(prop)
        elif attr["TYPE"] == "SCALAR":
            lookup["SCALAR"].append(prop)
        elif attr["TYPE"] == "VECTOR":
            lookup["VECTOR"].append(prop)
        else:
            lookup["MISCELLANEOUS"].append(prop)

    for category, props in lookup.items():
        if len(props) > 0:
            if doctype == "DOCSTRING":
                docs += "\n" f"{category} Properties\n" "========================\n"
                for prop in props:
                    attr = DATASET_PROPERTIES.get(prop, DATASET_PROPERTIES["_"])
                    docs += " ".join(
                        f"> \033[1;32;40m{prop}\033[0m ({attr['TYPE']}): {attr['DESC']}".split(".")[
                            :n
                        ]
                        + ["\n"]
                    )
            else:
                docs += (
                    f"""\n.. list-table:: **Table.** Available {category} Properties for Dataset.\n"""
                    """    :widths: 50 100           \n"""
                    """    :header-rows: 1           \n\n"""
                )
                docs += """    * - Property          \n""" """      - Description       \n"""
                for prop in props:
                    attr = DATASET_PROPERTIES.get(prop, DATASET_PROPERTIES["_"])
                    docs += f"""    * - ``{prop}``          \n""" f"""      - {attr['DESC']}  \n"""
    docs += "\n"
    return docs
