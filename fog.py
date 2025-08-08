import cv2
import numpy as np
import argparse
from pathlib import Path
from . import tool_kit as tk
from .parameter import const


def gen_fog(
    img_path: Path,
    depth_path: Path,
    output_path: Path,
    reduce_lum: int = 0,
    depth_multiplier: float = None,
    bar: bool = False,
) -> None:
    """
    Generate foggy image from rgb image and depth image
    :param img_path: rgb image path
    :param depth_path: depth image path
    :param output_path: output image path
    :param reduce_lum: factor defining the luminance reduction [0, 255]
    :param bar: whether to show the progress bar
    :param depth_multiplier: multiplier for the depthmap (default: None)
    :return: None
    """

    np.set_printoptions(threshold=np.inf)
    np.errstate(invalid="ignore", divide="ignore")

    # Load rgb and depth image
    img = cv2.imread(str(img_path))

    # Reduce the luminance of the image if needed
    if reduce_lum > 0:
        img[img >= reduce_lum] -= reduce_lum
        img[img < reduce_lum] = 0

    depth = cv2.imread(str(depth_path))[:, :, 0].astype(np.float64)
    depth[depth == 0] = 1  # the depth_min shouldn't be 0
    if depth_multiplier is not None:
        depth *= depth_multiplier

    I = np.empty_like(img)
    result = np.empty_like(img)

    elevation, distance, _ = tk.elevation_and_distance_estimation(
        img,
        depth,
        const.CAMERA_VERTICAL_FOV,
        const.HORIZONTAL_ANGLE,
        const.CAMERA_ALTITUDE,
        bar=bar,
    )

    if const.FT != 0:
        perlin = tk.noise(img, depth)
        ECA = const.ECA
        c = 1 - elevation / (const.FT + 0.00001)
        c[c < 0] = 0

        if const.FT > const.HT:
            ECM = (const.ECM * c + (1 - c) * const.ECA) * (perlin / 255)
        else:
            ECM = (const.ECA * c + (1 - c) * const.ECM) * (perlin / 255)
    else:
        ECA = const.ECA
        ECM = const.ECM

    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)

    if const.FT == 0:  # only haze: const.FT should be set to 0
        idx1 = elevation > const.HT
        idx2 = elevation <= const.HT

        if const.CAMERA_ALTITUDE <= const.HT:
            distance_through_haze[idx2] = distance[idx2]
            distance_through_haze_free[idx1] = (
                (elevation[idx1] - const.HT)
                * distance[idx1]
                / (elevation[idx1] - const.CAMERA_ALTITUDE)
            )

            distance_through_haze[idx1] = (
                distance[idx1] - distance_through_haze_free[idx1]
            )
        elif const.CAMERA_ALTITUDE > const.HT:
            distance_through_haze_free[idx1] = distance[idx1]
            distance_through_haze[idx2] = (
                (const.HT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze_free[idx2] = (
                distance[idx2] - distance_through_fog[idx2]
            )

        I[:, :, 0] = img[:, :, 0] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_haze_free
        )
        I[:, :, 1] = img[:, :, 1] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_haze_free
        )
        I[:, :, 2] = img[:, :, 2] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_haze_free
        )
        O = 1 - np.exp(
            -ECA * distance_through_haze - const.ECM * distance_through_haze_free
        )
    elif const.FT < const.HT and const.FT != 0:
        idx1 = np.logical_and(const.HT > elevation, elevation > const.FT)
        idx2 = elevation <= const.FT
        idx3 = elevation >= const.HT
        if const.CAMERA_ALTITUDE <= const.FT:
            distance_through_fog[idx2] = distance[idx2]
            distance_through_haze[idx1] = (
                (elevation[idx1] - const.FT)
                * distance[idx1]
                / (elevation[idx1] - const.CAMERA_ALTITUDE)
            )

            distance_through_fog[idx1] = distance[idx1] - distance_through_haze[idx1]
            distance_through_fog[idx3] = (
                (const.FT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx3] = (
                (const.HT - const.FT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze_free[idx3] = (
                distance[idx3]
                - distance_through_haze[idx3]
                - distance_through_fog[idx3]
            )
        elif const.CAMERA_ALTITUDE > const.HT:
            distance_through_haze_free[idx3] = distance[idx3]
            distance_through_haze[idx1] = (
                (const.FT - elevation[idx1])
                * distance_through_haze_free[idx1]
                / (const.CAMERA_ALTITUDE - const.HT)
            )
            distance_through_haze_free[idx1] = (
                distance[idx1] - distance_through_haze[idx1]
            )

            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = (
                (const.HT - const.FT)
                * distance
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze_free[idx2] = (
                distance[idx2]
                - distance_through_haze[idx2]
                - distance_through_fog[idx2]
            )
        elif const.FT < const.CAMERA_ALTITUDE <= const.HT:
            distance_through_haze[idx1] = distance[idx1]
            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
            distance_through_haze_free[idx3] = (
                (elevation[idx3] - const.HT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx3] = (
                (const.HT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )

        I[:, :, 0] = img[:, :, 0] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        I[:, :, 1] = img[:, :, 1] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        I[:, :, 2] = img[:, :, 2] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        O = 1 - np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)
    elif const.FT > const.HT:
        if const.CAMERA_ALTITUDE <= const.HT:
            idx1 = np.logical_and(const.FT > elevation, elevation > const.HT)
            idx2 = elevation <= const.HT
            idx3 = elevation >= const.FT

            distance_through_haze[idx2] = distance[idx2]
            distance_through_fog[idx1] = (
                (elevation[idx1] - const.HT)
                * distance[idx1]
                / (elevation[idx1] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
            distance_through_haze[idx3] = (
                (const.HT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_fog[idx3] = (
                (const.FT - const.HT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze_free[idx3] = (
                distance[idx3]
                - distance_through_haze[idx3]
                - distance_through_fog[idx3]
            )
        elif const.CAMERA_ALTITUDE > const.FT:
            idx1 = np.logical_and(const.HT > elevation, elevation > const.FT)
            idx2 = elevation <= const.FT
            idx3 = elevation >= const.HT

            distance_through_haze_free[idx3] = distance[idx3]
            distance_through_haze[idx1] = (
                (const.FT - elevation[idx1])
                * distance_through_haze_free[idx1]
                / (const.CAMERA_ALTITUDE - const.HT)
            )
            distance_through_haze_free[idx1] = (
                distance[idx1] - distance_through_haze[idx1]
            )
            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = (
                (const.HT - const.FT)
                * distance
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze_free[idx2] = (
                distance[idx2]
                - distance_through_haze[idx2]
                - distance_through_fog[idx2]
            )
        elif const.HT < const.CAMERA_ALTITUDE <= const.FT:
            idx1 = np.logical_and(const.HT > elevation, elevation > const.FT)
            idx2 = elevation <= const.FT
            idx3 = elevation >= const.HT

            distance_through_haze[idx1] = distance[idx1]
            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = distance[idx2] - distance_through_fog[idx2]
            distance_through_haze_free[idx3] = (
                (elevation[idx3] - const.HT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx3] = (
                (const.HT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )

        I[:, :, 0] = img[:, :, 0] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        I[:, :, 1] = img[:, :, 1] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        I[:, :, 2] = img[:, :, 2] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        O = 1 - np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)

    Ial = np.empty_like(img)  # color of the fog/haze
    Ial[:, :, 0] = 225
    Ial[:, :, 1] = 225
    Ial[:, :, 2] = 201

    result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]
    result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]
    result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]

    cv2.imwrite(str(output_path), result)


#### MAIN ####


# Create argument parser
parser = argparse.ArgumentParser(description="Generate fog data")


# Defien the parsing functions
def str2path(str: str) -> None:
    return Path(str)


# Add arguments
parser.add_argument(
    "-r", "--rgb", type=str2path, required=True, help="Path to input rgb file"
)
parser.add_argument(
    "-d", "--depth", type=str2path, required=True, help="Path to output depth file"
)
parser.add_argument(
    "-o",
    "--out",
    type=str2path,
    default=Path("./out.png)"),
    required=False,
    help="Path to output fog file",
)
parser.add_argument(
    "-l",
    "--reduce_lum",
    type=int,
    default=0,
    required=False,
    help="Reduce luminance by this amount",
)

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()

    # Call gen_fog function with arguments
    gen_fog(
        img_path=args.rgb,
        depth_path=args.depth,
        output_path=args.out,
        reduce_lum=args.reduce_lum,
    )
