"""
Main module for generating fog.

Constants:
    None

Raises:
    FileNotFoundError: If the input image or depth map is not found.
    ValueError: If the input image and depth map do not have the same dimensions.

Functions:
    gen_fog: Generate foggy image from rgb image and depth image

Classes:
    None

Author: Matteo Caligiuri
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from . import tool_kit as tk
from .parameter import const


def gen_fog(
    img_path: Path,
    depth_path: Path,
    output_path: Path,
    reduce_lum: int = 0,
    reduce_sat: int = 0,
    depth_multiplier: float = None,
    depth_flattening: bool = False,
    p_bar: bool = False,
) -> None:
    """
    Generate foggy image from rgb image and depth image.

    Args:
        img_path (Path): The path to the RGB image.
        depth_path (Path): The path to the depth image.
        output_path (Path): The path to the output image.
        reduce_lum (int, optional): The amount to reduce luminance.
            Defaults to 0.
        reduce_sat (int, optional): The amount to reduce saturation.
            Defaults to 0.
        depth_multiplier (float, optional): The multiplier for the depth map. Defaults to None.
        depth_flattening (bool): Whether to apply depth flattening.
            depth = 0.5 + (depth / 2)
            Defaults to False.
        p_bar (bool, optional): Whether to show a progress bar. Defaults to False.

    Returns:
        None
    """

    np.set_printoptions(threshold=np.inf)
    np.errstate(invalid="ignore", divide="ignore")

    # Load rgb and depth image
    img = cv2.imread(str(img_path))

    # Reduce the luminance of the image if needed
    if reduce_lum > 0 or reduce_sat > 0:
        # Convert the image to HSV as int
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)

        # Reduce the luminace by reduce_lum
        if reduce_sat > 0:
            img[:, :, 1] -= reduce_sat
        # Reduce the saturation by reduce_sat
        if reduce_lum > 0:
            img[:, :, 2] -= reduce_lum
        # Ensure that there is no < 0 value
        img[img < 0] = 0

        # Convert the image back to BGR as uint8
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    depth = cv2.imread(str(depth_path))[:, :, 0].astype(np.float64)
    depth[depth == 0] = 1  # the depth_min shouldn't be 0
    if depth_multiplier is not None:
        depth *= depth_multiplier

    if depth_flattening:
        depth = 0.5 + (depth / 2)

    I = np.empty_like(img)
    result = np.empty_like(img)

    elevation, distance, _ = tk.elevation_and_distance_estimation(
        img,
        depth,
        const.CAMERA_VERTICAL_FOV,
        const.HORIZONTAL_ANGLE,
        const.CAMERA_ALTITUDE,
        bar=p_bar,
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
    Ial[:, :, 2] = 225

    result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]
    result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]
    result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]

    cv2.imwrite(str(output_path), result)


#### MAIN ####
# Create argument parser
parser = argparse.ArgumentParser(description="Generate fog data")


# Define the parsing functions
def str2path(x: str) -> None:
    """
    Convert a string to a Path object.

    Args:
        x (str): The string to convert.

    Returns:
        Path: The converted Path object.
    """
    return Path(x)


# Add arguments
parser.add_argument(
    "-r", "--rgb", type=str2path, required=True, help="Path to input rgb file"
)
parser.add_argument(
    "-d", "--depth", type=str2path, required=True, help="Path to input depth file"
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
parser.add_argument(
    "-s",
    "--reduce_sat",
    type=int,
    default=0,
    required=False,
    help="Reduce saturation by this amount",
)
parser.add_argument(
    "-f",
    "--depth-flattening",
    action="store_true",
    default=False,
    required=False,
    help="Apply depth flattening",
)
parser.add_argument(
    "-m",
    "--depth-multiplier",
    type=float,
    default=None,
    required=False,
    help="Multiplier for fog intensity",
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
        reduce_sat=args.reduce_sat,
        depth_flattening=args.depth_flattening,
        depth_multiplier=args.depth_multiplier,
    )
