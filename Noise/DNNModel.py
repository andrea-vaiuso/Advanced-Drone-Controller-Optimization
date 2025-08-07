import numpy as np


class RotorSoundModel:
    def __init__(self, rpm_reference: float = 2500, rpm_exponent: float = 5.0,
                 filename: str = "angles_swl.npy"):
        """Load pre-computed SWL data and store reference parameters."""
        self.noise_data = np.load(filename, allow_pickle=True)
        self.rpm_reference = rpm_reference
        self.rpm_exponent = rpm_exponent

    def get_noise_emissions(self, zeta_angle, rpms, distance) -> tuple:
        """Return SPL and SWL for a given emission angle, RPMs and distance."""
        zeta_index = min(int(zeta_angle * 180 / np.pi), len(self.noise_data) - 1)
        # Reference SWL for one rotor (remove contribution of the other three)
        swl_ref_rpm = self.noise_data[zeta_index] - 6.02

        swl = self.total_swl_contribution(
            swl_ref_rpm, rpms, self.rpm_reference, self.rpm_exponent
        )
        # Free-field spherical spreading
        spl = swl - 10 * np.log10(4 * np.pi * (distance + 1e-4) ** 2)
        return spl, swl

    @staticmethod
    def total_swl_contribution(swl_ref_rpm, rpms, rpm_reference, rpm_exponent):
        """Combine SWL contributions from multiple rotors."""
        rpm_ratio = (np.array(rpms) + 1) / rpm_reference
        swl_individual = swl_ref_rpm + 10 * np.log10(rpm_ratio ** rpm_exponent)
        powers = 10 ** (swl_individual / 10)
        swl_total = 10 * np.log10(powers.sum())
        return swl_total

