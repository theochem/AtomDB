# Promolecular Densities and Properties

Promolecular properties can be computed for any molecule. 

The right way to compute a promolecular property has to do with whether the property is extensive or intensive. 

## Extensive Properties
For an extensive property, the promolecular estimate of the molecular property is given by a sum.
$$
p_{\text{mol;extensive}}^{(0)} = \sum_{A=1}^{N_{\text{atoms}}} c_A p_A^{(0)}
$$
where by default $c_A=1$. The atomic property is typically just that of the neutral atom, but when atomic populations/charges are known, it is appropriate to use the piecewise-linear approximation, and use
$$
p_A^{(0)} = \left(N_A - \lfloor N_A \rfloor\right) p_{A,\lceil N_A \rceil}^{(0)} + \left(\lceil N_A \rceil - N_A  \right) p_{A,\lfloor N_A \rfloor}^{(0)}
$$

## Intensive Properties
For an intensive property, the promolecular estimate is given by an average. In the literature, there are numerous different means used, typically geometric and arithmetic. We should investigate which works best for specified properties (to define certain defaults) but the user should always be able to choose their own. For most local properties, only the arithmetic mean makes sense. The [power mean](https://en.wikipedia.org/wiki/Generalized_mean) (it's a pending [pull request](https://github.com/scipy/scipy/pull/15729/files) in scipy) implements this,
$$
\langle x \rangle_p =  \left(\frac{1}{n}\sum_{k=1}^n x_k^p \right)^{\tfrac{1}{p}}
$$
For intensive properties, the appropriate solution is
$$
p_{\text{mol;intensive}}^{(0)} = \left \langle \left\{ p_A^{(0)} \right\}_{A=1}^{N_{\text{atoms}}} \right \rangle_p
$$

## Sub-Intensive Properties
We do not have many sub-intensive properties, which are properties that decrease to zero with the size of the system (like the chemical hardness for atomic clusters that become bulk metals). I wouldn't support sub-intensive properties. 

## Known Molecular (Local) Properties
However, in some cases, the molecular property is known. In that case you can imagine trying to determine the expansion coefficients to fit the data. For example, for some applications, it is important to try to match the value of the electron density (or other property) at/near the nuclei. In such cases, you can solve the least-squares system
$$
\left\{p_{\text{mol}}^{(0)}(\mathbf{r}_k) = \sum_{A=1}^{N_{\text{atoms}}} c_A p_A^{(0)}(\mathbf{r}_k) \right\}_{k=1}^{N_{\text{equations}} \ge N_{\text{atoms}}}
$$
to determine the expansion coefficients. This should be done with a method like `SLSQP` as one should constraint $c_A \ge 0$.
This workflow can work for extensive or intensive local properties.

## Implementation Notes
- one way to implement this is to allow the user to specify a value for $p$ in the p-mean, with `None` defaulting to extensive properties. A default value for each property should be defined too.

## List of properties with extensive/intensive classification
These are taken from [the issue](https://github.com/QuantumElephant/atomdb/issues/2)

### Intensive:
- Ionization potential. 
- Electron affinity. (same raw data as ionization potentials but accessed differently)
- chemical potentials (benchmark values via the Mulliken-Donnelly-Parr-Levy-Palke definition)
- hardness (benchmark values via the Parr-Pearson definition) [technically subintensive, but this only appears for conductors]
- spin-chemical potential vectors (both native and valence-state-averaged; both {N,S} and {N_alpha,N_beta} representations.)
- spin-hardness matrices (both native and valence-state averaged; both {N,S} and {N_alpha,N_beta} representations.)
- softness and spin-softness. (Technically extensive, but it only appears for conductors)
- Orbital energies (HF and/or DFT level). Utility routines for finding the HOMO/LUMO energy should be there, and could be considered "scalar properties" individually.

### Extensive:
- Electronic Energies (including HF, CI, experiment)
- polarizability 
- hyperpolarizability (if we can find it)
- Atomic density. (spline of values at specified points.)
- Core atomic density. (spline)
- Valence atomic density (spline)
- kinetic-energy density
- Kohn-Sham potential (wish list; perhaps far off except for certain special cases)
- exchange-correlation energy density (wish list; perhaps far off except for certain special cases)

### Neither (makes no sense in a promolecular context)
- Atomic radii (as many definitions as possible)
- fitting coefficients/exponents for Slaters to the density
- fitting coefficients/exponents for MBIS to the density
- fitting coefficients/exponents for Gaussians to the density

**Note:** For organic molecules, there are often better rules. For example, [Bosque, R., and Sales, J., J. Chem. Inf. Comput. Sci. 42, 1154, 2002] propose:

$$
α= 0.32 + 1.51#C + 0.17#H + 0.57#O + 1.05#N + 2.99#S + 2.48#P + 0.22#F + 2.16#Cl + 3.29#Br + 5.45#I
$$
