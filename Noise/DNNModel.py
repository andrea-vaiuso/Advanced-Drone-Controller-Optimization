import numpy as np

class RotorSoundModel():
    def __init__(self, rpm_reference = 2500, filename = "angles_swl.npy"):
        """
        Initialize the noise model by loading the noise model data from a npy file.
        """
        self.noise_data = np.load(filename, allow_pickle=True)
        self.rpm_reference = rpm_reference

    def get_noise_emissions(self, zeta_angle, rpms, distance) -> tuple:
        """
        Get the Sound Pressure Level (SPL) based on the zeta angle and RPM.
        The zeta angle is in radians, and the RPM is the rotor speed.

        Parameters:
            zeta_angle (float): The angle in radians between 0 and 2π calculated as arctan(height/distance).
            rpms (list): List of RPM values for the rotors.
            distance (float): The distance from the noise source in meters.
        Returns:
            tuple: (SPL, SWL) where SPL is the Sound Pressure Level and SWL is the Sound Power Level.
        """
        # Convert radians to degrees and ensure index is within bounds
        zeta_index = min(int(zeta_angle * 180 / np.pi), len(self.noise_data) - 1) 
        # Sound Power Level reference for the given zeta angle
        swl_ref_rpm = self.noise_data[zeta_index] 
        # Weighted Sound Power Level based on RPM
        swl = swl_ref_rpm + 10 * np.log10(np.average(rpms) / self.rpm_reference) 
        # Sound Pressure Level adjusted for distance
        spl = swl - abs(10 * np.log10(1/(4 * np.pi * ((distance+1e-4)**2)))) 
        return abs(spl), abs(swl)