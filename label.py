# Import required libraries
from datetime import datetime
import json
import os
import sys
import cv2
import numpy as np


def display_images(directory, scale_multiplier):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                image_files.append(os.path.join(root, file))

    # Dictionary to store annotations
    annotations = {}

    try:
        current_image_index = 0
        previous_image_index = -1
        current_slice_index = 0

        window_name = "Image Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # Mouse click callback function
        mouse_x, mouse_y = 0, 0

        def mouse_callback(event, x, y, flags, param):
            nonlocal \
                annotations, \
                current_image_index, \
                current_slice_index, \
                mouse_x, \
                mouse_y

            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing rectangle
                annotations.setdefault(image_files[current_image_index], {})
                annotations[image_files[current_image_index]].setdefault(
                    current_slice_index, {}
                )
                annotations[image_files[current_image_index]][current_slice_index][
                    "rectangle"
                ] = (x, y)

            if event == cv2.EVENT_MOUSEMOVE:
                # Update mouse position
                mouse_x, mouse_y = x, y

            elif event == cv2.EVENT_LBUTTONUP:
                # Finish drawing rectangle
                annotations[image_files[current_image_index]][current_slice_index][
                    "rectangle"
                ] += (x, y)

            elif event == cv2.EVENT_RBUTTONUP:
                # Add point of interest
                annotations.setdefault(image_files[current_image_index], {})
                annotations[image_files[current_image_index]].setdefault(
                    current_slice_index, {}
                )
                annotations[image_files[current_image_index]][
                    current_slice_index
                ].setdefault("points", [])
                annotations[image_files[current_image_index]][current_slice_index][
                    "points"
                ].append((x, y))

        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            if current_image_index != previous_image_index:
                image_file = image_files[current_image_index]
                image = np.load(image_file)
                previous_image_index = current_image_index

            slice = image[current_slice_index]
            scaled_image = cv2.resize(
                slice, None, fx=scale_multiplier, fy=scale_multiplier
            )

            # Convert to colour for drawing
            scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

            # Add relative path and slice at the bottom of the image
            relative_path = os.path.relpath(image_file, directory)
            text = f"Image: {relative_path} | Slice: {current_slice_index}"
            cv2.putText(
                scaled_image,
                text,
                (10, scaled_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw rectangle
            if (
                image_file in annotations
                and current_slice_index in annotations[image_file]
            ):
                rectangle = annotations[image_file][current_slice_index].get(
                    "rectangle"
                )
                if rectangle:
                    if len(rectangle) == 2:
                        # Use current mouse position as second point
                        cv2.rectangle(
                            scaled_image,
                            rectangle[:2],
                            (mouse_x, mouse_y),
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.rectangle(
                            scaled_image, rectangle[:2], rectangle[2:], (0, 255, 0), 2
                        )

            # Draw points
            if (
                image_file in annotations
                and current_slice_index in annotations[image_file]
            ):
                points = annotations[image_file][current_slice_index].get("points")
                if points:
                    for point in points:
                        cv2.circle(scaled_image, point, 3, (0, 0, 255), -1)

            cv2.imshow(window_name, scaled_image)

            key = cv2.waitKeyEx(1)
            if key == 27:  # ESC key
                break
            elif key == 2424832:  # Left arrow key
                current_image_index = (current_image_index - 1) % len(image_files)
                current_slice_index = 0
            elif key == 2555904:  # Right arrow key
                current_image_index = (current_image_index + 1) % len(image_files)
                current_slice_index = 0
            elif key == 2490368:  # Up arrow key
                current_slice_index = (current_slice_index + 1) % image.shape[0]
            elif key == 2621440:  # Down arrow key
                current_slice_index = (current_slice_index - 1) % image.shape[0]
            elif key == ord("c") or key == ord("C"):
                # Clear annotations for current slice
                if (
                    image_file in annotations
                    and current_slice_index in annotations[image_file]
                ):
                    del annotations[image_file][current_slice_index]

        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        cv2.destroyAllWindows()

    finally:
        # Rescale annotations to account for scale multiplier
        for image_file in annotations:
            for slice_index in annotations[image_file]:
                if "rectangle" in annotations[image_file][slice_index]:
                    annotations[image_file][slice_index]["rectangle"] = tuple(
                        int(x / scale_multiplier)
                        for x in annotations[image_file][slice_index]["rectangle"]
                    )
                if "points" in annotations[image_file][slice_index]:
                    annotations[image_file][slice_index]["points"] = [
                        tuple(int(x / scale_multiplier) for x in point)
                        for point in annotations[image_file][slice_index]["points"]
                    ]

        # Output annotations as JSON
        json_file = f"output_annotations/annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, "w") as f:
            json.dump(annotations, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python label.py <directory> [<scale_multiplier>]")
        sys.exit(1)

    directory = sys.argv[1]

    scale_multiplier = 1
    if len(sys.argv) == 3:
        scale_multiplier = int(sys.argv[2])

    display_images(directory, scale_multiplier)
