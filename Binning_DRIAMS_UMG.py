import os
import pandas as pd
import numpy as np
import re 


#defining dataset paths
datasets = ["./DRIAMS_A/DRIAMS-A", "./DRIAMS_D/DRIAMS-D", "./DRIAMS_B/DRIAMS-B", "./DRIAMS_C/DRIAMS-C", "./13911744/MS-UMG/MS-UMG"]
#we define the names for the output files
output_file_names = ["DriamsA", "DriamsD", "DriamsB", "DriamsC", "MS-UMG"]


def get_add_intensities(sample_bins, intensities):
    """
    Aggregate intensities per (digitized) bin and return the per-bin average
    intensities together with the array-of-arrays of sample bin indices.

    Parameters
    ----------
    sample_bins : ndarray-like (1D, integer)
        Result of np.digitize(...) or similar, giving for each peak the
        bin index into which that peak falls.
    intensities : ndarray-like (1D, numeric)
        Intensity values corresponding to each peak in the spectrum.
        Must be same length as sample_bins.

    Returns
    -------
    new_intentisities : list of floats
        For each unique bin index (in ascending order), the mean of intensities
        of peaks that fell into that bin.
    new_sample_bins : list of ndarray
        A list whose i-th element is the array of bin indices (from sample_bins)
        equal to the i-th unique bin value. (This mirrors the grouping used to
        compute the mean.)
    """

    # For each unique bin index i, select intensities where sample_bins == i and compute the mean.
    # The comprehension iterates over np.unique(sample_bins) which returns the unique bin indices
    # in sorted order. If a bin has no entries it won't appear (np.unique returns only present bins).
    new_intentisities = ([intensities[sample_bins == i ].mean() for i in np.unique(sample_bins)])

    # For each unique bin index i, create an array of the corresponding sample_bins entries.
    # Note: this returns arrays of the same value (i) repeated — effectively the grouped indices,
    # not the positions/locations of peaks. This may be surprising; often one expects the positions
    # (e.g., np.where(sample_bins == i)[0]) rather than the repeated bin values.
    new_sample_bins = ([sample_bins[sample_bins == i ] for i in np.unique(sample_bins)])  

    # Return the two lists: per-bin means and grouped bin arrays.
    return new_intentisities , new_sample_bins



def process_binned_data(ms_data, ms_codes, driams_dataset, columns_headers):
    """
    Convert raw mass spectra into a binned intensity matrix, merge with metadata,
    and return the final DataFrame.

    Parameters
    ----------
    ms_data : list of DataFrames
        Each DataFrame contains 'mass' and 'intensity' columns of a sample.
    ms_codes : list
        Sample identifiers corresponding to each entry in ms_data.
    driams_dataset : DataFrame
        Metadata table containing the 'code' column used for merging.
    columns_headers : array-like
        The m/z bin edges used as feature columns.

    Returns
    -------
    DataFrame
        Binned intensity table merged with metadata.
    """

    # Create an empty matrix for storing binned spectra
    matrix = np.zeros((len(ms_data), len(columns_headers + 1)))

    # Loop through each spectrum
    for index, s in enumerate(ms_data):

        # Assign each mass value to a bin index
        indices_bins = np.digitize(s['mass'], columns_headers)

        # Sum intensities for all peaks that fall inside each bin
        add_intensities, new_sample_bins = get_add_intensities(
            indices_bins, 
            s['intensity'].values
        )

        # Fill the matrix row with summed intensities
        for i, value in enumerate(new_sample_bins):
            matrix[index][value - 1] = add_intensities[i]

    # Convert matrix to DataFrame with bin headers
    bin_driams_dataset = pd.DataFrame(matrix, columns=columns_headers)

    # Add sample codes
    bin_driams_dataset['code'] = ms_codes

    # Merge binned data with original metadata
    bin_driams_dataset = bin_driams_dataset.merge(driams_dataset, on='code')

    return bin_driams_dataset




# MAIN PROCESSING PIPELINE FOR ALL DATASETS
for j, setss in enumerate(datasets):

    id_folder = os.path.join(setss, "id")
    raw_mass_spectra = os.path.join(setss, "preprocessed")

    # Get all year folders (only digits)
    year_folders = sorted([folder for folder in os.listdir(id_folder) if folder.isdigit()])


    # Process each year separately
    for year in year_folders:

        id_file_path = os.path.join(id_folder, year, f"{year}_clean.csv")
        raw_folder_path = os.path.join(raw_mass_spectra, year)
        raw_file_path = [files for files in os.listdir(raw_folder_path)]

        # Define m/z binning parameters
        ms_min, ms_max, ms_bin_size = 2000, 20000, 3

        # Lists to collect raw spectra
        ms_codes, ms_data = [], []

        # Load metadata file
        driams_dataset = pd.read_csv(id_file_path)

        # SPECIAL PROCESSING FOR DRIAMS-A AND DRIAMS-D (split into 4 parts)
        if setss == "./DRIAMS_A/DRIAMS-A" or setss == "./DRIAMS_D/DRIAMS-D":
            
            # Split dataset to reduce memory usage
            parts = np.array_split(driams_dataset, 4)
            bin_driams_datasets = []

            for part in parts:

                temp_ms_data, temp_ms_codes = [], []

                # Load only spectra whose codes belong to this split
                for file_name in raw_file_path:
                    aux = re.sub('.txt', '', file_name)
                    if aux in part['code'].values:
                        mass_df = pd.read_csv(
                            os.path.join(raw_folder_path, file_name),
                            names=['mass', 'intensity'],
                            sep=' ',
                            skiprows=3
                        )
                        temp_ms_data.append(mass_df)
                        temp_ms_codes.append(aux)

                # Compute bin borders
                read_mass = np.concatenate([df['mass'].values for df in temp_ms_data])
                bins = np.arange(ms_min, ms_max, ms_bin_size)
                columns_headers = np.sort(bins[~np.isnan(bins)]).astype(int)

                # Convert raw spectra → bin matrix
                bin_driams_datasets.append(
                    process_binned_data(temp_ms_data, temp_ms_codes, part, columns_headers)
                )

            # Combine four split parts back together
            bin_driams_dataset = pd.concat(bin_driams_datasets, ignore_index=True)
            print("done")


        # NORMAL PROCESSING FOR OTHER DATASETS (C, B, UMG, ...)
        else:
            for file_name in raw_file_path:
                aux = re.sub('.txt', '.txt', file_name)

                # Keep only files with codes present in metadata
                if aux in driams_dataset['code'].values:
                    mass_df = pd.read_csv(
                        os.path.join(raw_folder_path, file_name),
                        names=['mass', 'intensity'],
                        sep=' ',
                        skiprows=0
                    )
                    ms_data.append(mass_df)
                    ms_codes.append(aux)

            # Create bin headers
            read_mass = np.concatenate([df['mass'].values for df in ms_data])
            bins = np.arange(ms_min, ms_max, ms_bin_size)
            columns_headers = np.sort(bins[~np.isnan(bins)]).astype(int)

            # Convert whole dataset to bins
            bin_driams_dataset = process_binned_data(
                ms_data, ms_codes, driams_dataset, columns_headers
            )

        # SAVE THE FINAL BINNED CSV
        file_name = f'{output_file_names[j]}_{year}_bin3_2000-20000.csv'
        bin_driams_dataset.to_csv(file_name, sep=',', index=False)
