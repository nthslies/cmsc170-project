# imports
import tkinter as tk
from tkinter.filedialog import askopenfilename
import zipfile
import csv
import io
import re
import numpy as np
import math

# function to load the data
def read_collection():
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
    root.destroy()

    # if no file selected
    if not zip_path:
        print("No ZIP file selected.")
        return None
    print(f"{zip_path} successfully loaded!")

    # for storing monthly collections
    collections = {}

    # maps month to equivalent number
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    try:
        # opening a zip file
        with zipfile.ZipFile(zip_path, 'r') as zip:
            for filename in sorted(zip.namelist()):
                if 'bir_collection' in filename.lower() and filename.lower().endswith('.csv'):
                    # for extracting the year from filename
                    year_match = re.search(r'(202[0-4])', filename)
                    if not year_match:
                        continue
                    year = int(year_match.group())

                    # opening each file in zip
                    with zip.open(filename) as f:
                        text = io.TextIOWrapper(f, encoding = 'utf-8-sig')
                        reader = csv.DictReader(text)
                        # ensures header exists
                        header = reader.fieldnames
                        if not header:
                            continue
                        # cleaning the headers
                        cleaned_header = {h: h.strip().lower() for h in header if h} 
                        # gets regions
                        region_column = None
                        for h in header:
                            if  h and h.strip().lower() == 'region':
                                region_column = h
                                break

                        # maps month to its equivalent number
                        active_month = {origin: months[clean] for origin, clean in cleaned_header.items() if clean in months}                    
                        # ensures required columns exists
                        if len(active_month) != 12 or not region_column:
                            continue

                        # parse each row
                        for row in reader:
                            region = row.get(region_column, '')
                            if region is None:
                                continue
                            region = region.strip()
                            if not region or 'total' in region.lower():
                                continue

                            # loops in each active month
                            for column_name, month_number in active_month.items():
                                raw_value = row.get(column_name, '')
                                if not raw_value:
                                    continue
                                cleaned_value = str(raw_value).replace(',', '').strip()

                                if not cleaned_value or cleaned_value == "-" or cleaned_value.lower() == "nan" or cleaned_value.lower() == "n/a":
                                    continue

                                try:
                                    revenue = float(cleaned_value)
                                    # skips nan and infinity values
                                    if not math.isfinite(revenue):
                                        continue
                                    key = (region, month_number, year)
                                    if key not in collections:
                                        collections[key] = 0.0
                                    # adds collections from same region
                                    collections[key] += revenue
                                except ValueError:
                                    continue

    except (zipfile.BadZipFile, OSError) as error:
        print(f"Error opening ZIP file: {error}")
        return None

    if not collections:
        print("No valid collection data found.")
        return None

    return collections

def load_data():
    collections = read_collection()
    if collections is None:
        return None, None, None, None
    # sorting the regions
    sorted_region = sorted(set(region for region, month, year in collections))
    # maps region to index
    region_to_id = {region: index for index, region in enumerate(sorted_region)}
    # initializes train and test data
    x_train, y_train, x_test, y_test = [], [], [], []
    # loops in collections
    for (region, month, year) in sorted(collections):
        current_revenue = collections[(region, month, year)]
        previous_year = year - 1
        # makes sure data exists for specific month and region
        if (region, month, previous_year) in collections:
            previous_revenue = collections[(region, month, previous_year)]
            # compares current with previous
            if current_revenue > previous_revenue:
                label = 1 
            else:
                label = 0
            region_features = [0.0] * len(sorted_region) # creates list for all regions 
            region_idx = region_to_id[region] # index of current
            region_features[region_idx] = 1.0 # set current to 1
            # to represent months as yearly cycle
            month_sin = math.sin( 2 * math.pi * month / 12)
            month_cos = math.cos( 2 * math.pi * month / 12)
            features = region_features + [month_sin, month_cos, previous_revenue]
            # for identifying if training or test data
            if 2021 <= year <= 2023:
                x_train.append(features)
                y_train.append(label)
            elif year == 2024:
                x_test.append(features)
                y_test.append(label)

    # checks if train or test data is empty
    if not x_train or not x_test:
        print("Training or testing data is empty.")
        return None, None, None, None
    # converts to arrays
    x_train = np.array(x_train, dtype = float)
    y_train = np.array(y_train, dtype = int)
    x_test = np.array(x_test, dtype = float)
    y_test = np.array(y_test, dtype = int)
    # prints info about dataset
    print(f"\nNumber of Regions: {len(sorted_region)}")
    print(f"Training Records (2021-2023): {len(x_train)}")
    print(f"Testing Records (2024): {len(x_test)}")
    print(f"Number of Features: {x_train.shape[1]}")
    
    return(x_train, y_train, x_test, y_test)

def load_clustering():
    collections = read_collection()
    if collections is None:
        return None, None
    # sorting the regions
    sorted_region = sorted(set(region for region, month, year in collections))
    # gets all regions
    x_regions = []
    region_names = []
    for region in sorted_region:
        monthly_values = []
        yearly_totals = {}
        # collects values from each month from 2020 to 2024
        for year in range(2020, 2025):
            year_values = []
            for month in range(1, 13):
                key = (region, month, year)
                if key in collections:
                    revenue = collections[key]
                    monthly_values.append(revenue)
                    year_values.append(revenue)
            # ensures all 12 months exists for getting sum of year
            if len(year_values) == 12:
                yearly_totals[year] = sum(year_values)

        # ensures region has complete data
        if len(monthly_values) != 60 or len(yearly_totals) != 5:
            print(f"Skipped {region}: incomplete collection history.")
            continue

        mean_monthly_collection = np.mean(monthly_values) # means
        growth_rates = [] # annual growth rate
        for year in range(2021, 2025):
            previous_total = yearly_totals[year - 1]
            current_total = yearly_totals[year]
            # handles division by zero
            if previous_total != 0:
                growth = ((current_total - previous_total) / previous_total)
                growth_rates.append(growth)

        # ensures theres 4 growth rates
        if len(growth_rates) != 4:
            print(f"Skipped {region}: growth rate unavailable.")
            continue
        average_annual_growth_rate = np.mean(growth_rates)

        if mean_monthly_collection == 0:
            print(f"Skipped {region}: zero mean collection.")
            continue
        collection_volatility = (np.std(monthly_values) / abs(mean_monthly_collection))
        # combine clustering features
        features = [mean_monthly_collection, average_annual_growth_rate, collection_volatility]
        x_regions.append(features)
        region_names.append(region)

    if not x_regions:
        print("No valid clustering data found.")
        return None, None
    # converts to array
    x_regions = np.array(x_regions, dtype = float)
    # prints info
    print(f"Number of Regions: {len(region_names)}")
    print(f"Number of Features: {x_regions.shape[1]}")

    return x_regions, region_names

if __name__ == "__main__":
    # supervised
    x_train, y_train, x_test, y_test = load_data()
    if x_train is not None:
        # indicator that data loaded
        print("\nData Loading Completed!")

    # unsupervised
    x_regions, region_names = load_clustering()
    if x_regions is not None:
        # indicator that data loaded
        print("\nData Loading Completed!")