import csv
import sys
csv.field_size_limit(sys.maxsize)
import ast
import os
import json
import argparse
import re

from imageCaptioning.config import DATA_ROOT, LOGGER

def prepareFromCSV(csvPath : str, split : str, outputDirectory : str = DATA_ROOT):
    """
    Convert CSV Data with Bytes Info into:
        1. Images / Directory of JPG Files
        2. annotations.json with the metadata for the preprocessing
    """
    imageDirectory = os.path.join(outputDirectory, "images", split)
    os.makedirs(imageDirectory, exist_ok=True)

    annotations = []
    LOGGER.info(f"Reading RSICD CSV from {csvPath} ...")

    with open(csvPath, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if not row or len(row) < 3:
                continue
            filename = row[0]
            captionString = row[1]
            byteString = row[2]
            # Parse the captions
            try:
                captionsList = ast.literal_eval(captionString)
                captionsList = [str(c).strip() for c in captionsList]

                if len(captionsList) == 1:
                    joined = captionsList[0]
                    joined = re.sub(r'\.(\w)', r'. \1', joined)
                    splitCaptions = re.split(r'\.\s+', joined)
                    captionsList = [c.strip() for c in splitCaptions if c.strip()]
            except Exception:
                captionsList = captionString.strip("[]").replace("'", "").split(",")
                captionsList = [c.strip() for c in captionsList if c.strip()]

            # Parse the Image Bytes
            try:
                dictObject = ast.literal_eval(byteString)
                imageBytes = dictObject["bytes"]
            except Exception as e:
                LOGGER.warning(f"Row {index}: Failed to parse image bytes ({e})")
                continue

            # Save the image
            basename = os.path.basename(filename)
            outputPath = os.path.join(imageDirectory, basename)
            with open(outputPath, "wb") as image:
                image.write(imageBytes)

            # Collect the metadata
            annotations.append({
                "imageId" : os.path.splitext(basename)[0],
                "filename" : basename,
                "captions" : captionsList,
            })
        
        # Save annotations to the json file
        annotationPath = os.path.join(outputDirectory, f"{split}Annotations.json")
        with open(annotationPath, "w") as jsonFile:
            json.dump(annotations, jsonFile, indent=2)
        
        LOGGER.info(f"Saved {len(annotations)} images : {imageDirectory}")
        LOGGER.info(f"Saved Metadata : {annotationPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the RSICD dataset from the CSV file")
    parser.add_argument("--csv",
                        type=str,
                        required=True,
                        help="Path to the RSICD CSV files")
    parser.add_argument("--output",
                        type=str,
                        default=DATA_ROOT,
                        help="Output Directory (default : DATA_ROOT)")
    parser.add_argument("--split",
                        type=str,
                        choices=["train", "test", "valid"],
                        required=True,
                        help="Split type of the image data")
    arguments = parser.parse_args()

    prepareFromCSV(arguments.csv, arguments.split, arguments.output)

