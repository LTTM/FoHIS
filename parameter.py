"""
Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
Ning Zhang, Lin Zhang*, and Zaixi Cheng
"""


from . import const


# atmosphere
const.VISIBILITY_RANGE_MOLECULE = 12  # m
const.VISIBILITY_RANGE_AEROSOL = 450  # m
const.ECM = 3.912 / const.VISIBILITY_RANGE_MOLECULE  # EXTINCTION_COEFFICIENT_MOLECULE /m
const.ECA = 3.912 / const.VISIBILITY_RANGE_AEROSOL  # EXTINCTION_COEFFICIENT_AEROSOL /m

const.FT = 31  # FOG_TOP m
const.HT = 300  # HAZE_TOP m

# camera
const.CAMERA_ALTITUDE = 50  #  m
const.HORIZONTAL_ANGLE = 0  #  °
const.CAMERA_VERTICAL_FOV = 64  #  °
