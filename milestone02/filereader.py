# imports
import tkinter as tk
from tkinter.filedialog import askopenfilename
import zipfile
import csv
import io
import re
import numpy as np

# function to load the data
def load_data():
    # initialize window in order to open file
    root = tk.Tk()
    root.withdraw()
    zip_path = askopenfilename(
        title = "Select BIR Tax Collection ZIP File",
        filetypes = [
            ("ZIP files", "*.zip"),
            ("All files", "*.*")
        ]
    )

    # if no file selected
    if not zip_path:
        print("No ZIP file selected.")
        return None, None, None, None
    print(f"{zip_path} successfully loaded!")

    # for storing monthly collections
    collections = {}
    regions_set = set()

    # maps month to equivalent number
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    # opening a zip file
    with zipfile.ZipFile(zip_path, 'r') as zip:
        for filename in zip.namelist():
            if 'bir_collection' in filename.lower() and filename.endswith('.csv'):
                # for extracting the year from filename
                year_match = re.search(r'(202[0-4])', filename)
                if not year_match:
                    continue
                year = int(year_match.group(1))

                # opening each file in zip
                with zip.open(filename) as f:
                    text = io.TextIOWrapper(f, encoding = 'utf-8')
                    reader = csv.DictReader(text)
                    header = reader.fieldnames
                    # cleaning the headers
                    cleaned_header = {h: h.strip().lower() for h in header if h} 
                    # gets regions
                    region_column = None
                    for h in header:
                        if 'region' in h.lower() or 'particulars' in h.lower():
                            region_column = h
                            break

                    # maps month to its equivalent number
                    active_month = {origin: months[clean] for origin, clean in cleaned_header.items() if clean in months}
                    if not active_month or not region_column:
                        continue

                    # parse each row
                    for row in reader:
                        region = row.get(region_column, '').strip()
                        if not region or 'total' in region.lower():
                            continue
                        regions_set.add(region)

                        # loops in each active month
                        for column_name, month_number in active_month.items():
                            raw_value = row.get(column_name, '')
                            if not raw_value:
                                continue
                            cleaned_value = str(raw_value).replace(',', '').strip()

                            if not cleaned_value or cleaned_value == "-" or cleaned_value.lower() == "nan":
                                continue
                            try:
                                revenue = float(cleaned_value)
                                # store in collections
                                collections[(region, month_number, year)] = revenue
                            except ValueError:
                                continue

    # sorting the regions
    sorted_region = sorted(list(regions_set))
    region_to_id = {region: index for index, region in enumerate(sorted_region)}

    x_train, y_train, x_test, y_test = [], [], [], []
    for (region, month, year), current_revenue in collections.items():
        previous_year = year - 1
        # makes sure data exists for specific month and region
        if (region, month, previous_year) in collections:
            previous_revenue = collections[(region, month, previous_year)]
            # compares current with previous
            if current_revenue > previous_revenue:
                label = 1 
            else:
                label = 0
            region_id = float(region_to_id[region])
            features = [region_id, float(month), previous_revenue]

            # for identifying if training or test data
            if 2021 <= year <= 2023:
                x_train.append(features)
                y_train.append(label)
            elif year == 2024:
                x_test.append(features)
                y_test.append(label)

    # converts to arrays
    return(
        np.array(x_train, dtype = float),
        np.array(y_train, dtype = int),
        np.array(x_test, dtype = float),
        np.array(y_test, dtype = int),
    )

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    if x_train is not None:
        # indicator that data loaded
        print(f"\nData Loading Completed!")