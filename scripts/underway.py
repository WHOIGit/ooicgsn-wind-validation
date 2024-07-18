import os
import numpy as np
import pandas as pd
import xarray as xr

ATTRS = {
    "latitude": {
        "long_name": 'Latitude',
        'standard_name': 'latitude',
        'units': 'degrees' },
    'longitude': {
        'long_name': 'Longitude',
        'standard_name': 'longitude',
        'units': 'degrees' },
    'ship_speed': {
        'long_name': 'Ship Speed',
        'units': 'kts' },
    'heading' : {
        'long_name': 'Heading',
        'units': 'degrees' },
    'course_over_ground': {
        'long_name': 'Course over Ground',
        'units': 'degrees',
        'comment': 'Ship course over ground from GPS.' },
    'speed_over_ground': {
        'long_name': 'Speed over Ground',
        'units': 'kts',
        'comment': 'Ship speed over ground from GPS.' },
    'air_temperature_port': {
        'standard_name': 'air_temperature',
        'long_name': 'Bulk Air Temperature - Port',
        'units': 'degrees_Celcius',
        'comment': ('Air Temperature refers to the temperature of the air surrounding the sensor; this is also referred to as bulk temperature. '
                    'This measurement originates from the Port side of the ship met sensor array. It has not been adjusted for height.') },
    'air_temperature_starboard': {
        'standard_name': 'air_temperature',
        'long_name': 'Bulk Air Temperature - Starboard',
        'units': 'degrees_Celcius',
        'comment': ('Air Temperature refers to the temperature of the air surrounding the sensor; this is also referred to as bulk temperature. '
                    'This measurement originates from the starboard side of the ship met sensor array. It has not been adjusted for height.') },
    'air_pressure_port': {
        'standard_name': 'air_pressure',
        'long_name': 'Air Pressure - Port',
        'units': 'hPa',
        'comment': ('Air Pressure is a measure of the force per unit area of the column of air above the sensor. This measurement originates from '
                    'the port side of the ship met sensor array. It has not been adjusted for height.') },
    'air_pressure_starboard': {
        'standard_name': 'air_pressure',
        'long_name': 'Air Pressure - Starboard',
        'units': 'hPa',
        'comment': ('Air Pressure is a measure of the force per unit area of the column of air above the sensor. This measurement originates from '
                    'the starboard side of the ship met sensor array. It has not been adjusted for height.') },
    'precipitation_rate_port': {
        'standard_name': 'precipitation_rate',
        'long_name': 'Precipitation Rate - Port',
        'units': 'mm/hr',
        'comment': ('Precipitation rate is a measure of the thickness of a layer of water that accumulates per unit time. '
                    'This measurement originates from the port side of the ship met sensor array.' ) },
    'precipitation_rate_starboard': {
        'standard_name': 'precipitation_rate',
        'long_name': 'Precipitation Rate - Starboard',
        'units': 'mm/hr',
        'comment': ('Precipitation rate is a measure of the thickness of a layer of water that accumulates per unit time. '
                    'This measurement originates from the starboard side of the ship met sensor array.' ) },
    'precipitation_amount_port': {
        'standard_name': 'thickness_of_precipitation_amount',
        'long_name': 'Amount of Precipitation - Port',
        'units': 'mm',
        'comment' : ('Precipitation amount is a measure of the thickness of the total water that accumulates. '
                     'This measurement originates from the port side of the ship met sensor array.') },
    'precipitation_amount_starboard': {
        'standard_name': 'thickness_of_precipitation_amount',
        'long_name': 'Amount of Precipitation - Starboard',
        'units': 'mm',
        'comment' : ('Precipitation amount is a measure of the thickness of the total water that accumulates. '
                     'This measurement originates from the starboard side of the ship met sensor array.') },
    'relative_wind_direction_port': {
        'standard_name': 'wind_to_direction',
        'long_name': 'Relative Wind Direction - Port',
        'units': 'degrees',
        'comment': ('The relative direction of the wind is given as the direction to which the wind is blowing, '
                    'uncorrected for ship motion. This measurement originates from the port side of the ship '
                    'met sensor array.') },
    'relative_wind_direction_starboard': {
        'standard_name': 'wind_to_direction',
        'long_name': 'Relative Wind Direction - Starboard',
        'units': 'degrees',
        'comment': ('The relative direction of the wind is given as the direction to which the wind is blowing, '
                    'uncorrected for ship motion. This measurement originates from the starboard side of the ship '
                    'met sensor array.') },    
    'relative_wind_speed_port': {
        'standard_name': 'wind_speed',
        'long_name': 'Relative Wind Speed - Port',
        'units': 'm s-1',
        'comments': ('Relative wind speed is the magnitude of the velocity, uncorrected for ship motion. '
                     'This measurements originates from the port side of the ship met sensor array.') },
    'relative_wind_speed_starboard': {
        'standard_name': 'wind_speed',
        'long_name': 'Relative Wind Speed - Starboard',
        'units': 'm s-1',
        'comments': ('Relative wind speed is the magnitude of the velocity, uncorrected for ship motion. '
                     'This measurements originates from the starboard side of the ship met sensor array.') },
    'relative_humidity_port': {
        'standard_name': 'relative_humidity',
        'long_name': 'Relative Humidity - Port',
        'units': 'percent',
        'comments': ('This measurements originates from the port side of the ship met sensor array.') },
    'relative_humidity_starboard': {
        'standard_name': 'relative_humidity',
        'long_name': 'Relative Humidity - Starboard',
        'units': 'percent',
        'comments': ('This measurements originates from the starboard side of the ship met sensor array.') },
    'wind_speed_port': {
        'standard_name': 'wind_speed',
        'long_name': 'Wind Speed - Port',
        'units': 'm s-1',
        'comment': ('Wind speed is the magnitude of the velocity, corrected for ship motion. '
                    'This measurement originates from the port side of the ship met sensor array.') },
    'wind_speed_starboard': {
        'standard_name': 'wind_speed',
        'long_name': 'Wind Speed - Starboard',
        'units': 'm s-1',
        'comment': ('Wind speed is the magnitude of the velocity, corrected for ship motion. '
                    'This measurement originates from the starboard side of the ship met sensor array.') },   
    'wind_direction_port': {
        'standard_name': 'wind_to_direction',
        'long_name': ' Wind Direction - Port',
        'units': 'degrees',
        'comment':('The direction of the wind is given as the direction to which the wind is blowing, '
                    'corrected for ship motion and relative to magnetic north. This measurement originates '
                    'from the port side of the ship met sensor array.') },
    'wind_direction_starboard': {
        'standard_name': 'wind_to_direction',
        'long_name': 'Wind Direction - Starboard',
        'units': 'degrees',
        'comment': ('The direction of the wind is given as the direction to which the wind is blowing, '
                    'corrected for ship motion and relative to magnetic north. This measurement originates '
                    'from the starboard side of the ship met sensor array.') }, 
    'barometric_pressure_port': {
        'standard_name': 'air_pressure',
        'long_name': 'Barometric Pressure - Port',
        'units': 'hPa',
        'comment': ('Barometric Pressure is a measure of the weight of the column of air above the sensor. '
                    'It is also commonly referred to as atmospheric pressure. This measurement originates '
                    'from the port side of the ship met sensor array and is adjusted to the height of the '
                    'sensor array.') },
    'barometric_pressure_starboard': {
        'standard_name': 'air_pressure',
        'long_name': 'Barometric Pressure - Starboard',
        'units': 'hPa',
        'comment': ('Barometric Pressure is a measure of the weight of the column of air above the sensor. '
                    'It is also commonly referred to as atmospheric pressure. This measurement originates '
                    'from the starboard side of the ship met sensor array and is adjusted to the height of the '
                    'sensor array.') },
    'shortwave_radiation': {
        'standard_name': 'downwelling_shortwave_flux_in_air',
        'long_name': 'Downwelling Shortwave Flux',
        'units': 'W m-2',
        'comment': ('Downwelling radiation is radiation from above with positive sign downwards. It does not mean "net downward".') },
    'longwave_radiation': {
        'standard_name': 'downwelling_longwave_flux_in_air',
        'long_name': 'Downwelling Longwave Flux',
        'units': 'W m-2',
        'comment': ('Downwelling radiation is radiation from above with positive sign downwards. It does not mean "net downward".') },
    'par': {
        'long_name': 'Photosynthetically Active Radiation',
        'units': 'uE m-2 s-1',
        'comment': ('Photosynthetically Active Radiation (PAR) is light of wavelengths 400-700 nm and is the portion of the light '
                    'spectrum utilized for photosynthesis. It is measured as photon flux density, which is the rate that umol of quanta '
                    'of light land per unit area.') },
    'sea_surface_salinity': {
        'standard_name': 'sea_surface_salinity',
        'long_name': 'Sea Surface Practical Salinity',
        'units': 'psu',
        'comment' : ('Salinity is generally defined as the concentration of dissolved salt in a parcel of seawater. Practical Salinity is '
                     'a more specific unitless quantity calculated from the conductivity of seawater and adjusted for temperature and pressure. '
                     'It is approximately equivalent to Absolute Salinity (the mass fraction of dissolved salt in seawater) but they are not '
                     'interchangeable. This measurement is made at 5m below the ship water line.') },
    'sea_surface_temperature': {
        'standard_name': 'sea_surface_temperature',
        'long_name': 'Sea Surface Temperature',
        'units': 'degrees_Celcius',
        'comment': ('Sea Surface Temperature is the temperature of the seawater near the ocean surface. This measurement is made at 5m below the '
                    'ship water line.') },
    'speed_of_sound': {
        'standard_name': 'speed_of_sound_in_seawater',
        'long_name': 'Speed of Sound in Water',
        'units': 'm s-1',
        'comment': ('This is the magnitude of the velocity of sound in the sea surface water.') },
    'depth12': {
        'long_name': 'Bottom Depth - 12kHz',
        'units': 'm',
        'comment': ('This is the calculated bottom depth from the 12kHz acoustic sensor.') },
    'depth35': {
        'long_name': 'Bottom Depth - 3.5kHz',
        'units': 'm',
        'comment': ('This is the calculated bottom depth from the 3.5kHz acoustic sensor.') },
    'em122': {
        'long_name': 'Bottom Depth - 12kHz multibeam center',
        'units': 'm',
        'comment': ('This is the calculated bottom depth from the 12kHz multibeam center return.') }
}

name_map = {
    'Dec_LAT': 'latitude',
    'Dec_LON': 'longitude',
    'SPD': 'ship_speed',
    'HDT': 'heading',
    'COG': 'course_over_ground',
    'SOG': 'speed_over_ground',
    'WXTP_Ta': 'air_temperature_port',
    'WXTS_Ta': 'air_temperature_starboard',
    'WXTP_Pa': 'air_pressure_port',
    'WXTS_Pa': 'air_pressure_starboard',
    'WXTP_Ri': 'precipitation_rate_port',
    'WXTS_Ri': 'precipitation_rate_starboard',
    'WXTP_Rc': 'precipitation_amount_port',
    'WXTS_Rc': 'precipitation_amount_starboard',
    'WXTP_Dm': 'relative_wind_direction_port',
    'WXTS_Dm': 'relative_wind_direction_starboard',
    'WXTP_Sm': 'relative_wind_speed_port',
    'WXTS_Sm': 'relative_wind_speed_starboard',
    'WXTP_Ua': 'relative_humidity_port',
    'WXTS_Ua': 'relative_humidity_starboard',
    'WXTP_TS': 'wind_speed_port',
    'WXTS_TS': 'wind_speed_starboard',
    'WXTP_TD': 'wind_direction_port',
    'WXTS_TD': 'wind_direction_starboard',
    'BAROM_P': 'barometric_pressure_port',
    'BAROM_S': 'barometric_pressure_starboard',
    'RAD_SW': 'shortwave_radiation',
    'RAD_LW': 'longwave_radiation',
    'PAR': 'par',
    'SBE45S': 'sea_surface_salinity',
    'SBE48T': 'sea_surface_temperature',
    'SSVdslog': 'speed_of_sound',
    'Depth12': 'depth12',
    'Depth35': 'depth35',
    'EM122': 'em122'
}

def parse_data_files(file_dir):
    """Parses the underway .csv files into a dataframe"""
    df = pd.DataFrame()
    for file in file_dir:
        if file.endswith(".csv"):
            data = pd.read_csv(file, header=1)
            df = pd.concat([df, data], ignore_index=True)

    # Clean up the dataframe
    for col in df:
        df.rename(columns={col: col.strip()}, inplace=True)
        
    # Convert the time and date into a single datetime column
    df["TIME_GMT"] = df["TIME_GMT"].apply(lambda x: x.replace(':60.000',':59.999'))
    df["time"] = df["DATE_GMT"] + df["TIME_GMT"]
    df["time"] = df["time"].apply(lambda x: pd.to_datetime(x))
    
    # Sort values
    df = df.sort_values(by=["time"])

    # Reindex
    df = df.set_index(keys=["time"], drop=True)

    # Drop unwanted variables
    drop_cols = ["DATE_GMT", "TIME_GMT", 'FLR', 'FLOW']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)
            
    # Replace NANs and NODATA with nans
    df = df.replace(' NAN', np.nan).replace(' NODATA', np.nan)
    
    return df


def parse_par_header(file, ATTRS):
    """Parse the R/V Armstrong PAR Met Sensor"""
    with open(file) as f:
        data = f.readlines()
        
    for line in data:
        # Get the make and model
        if "make" in line.lower() and "model" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["par"].update({key: value})
        elif "calibration date" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["par"].update({key: value})
        elif "s/n" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["par"].update({key: value})
        elif "installation" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["par"].update({key: value})
        else:
            pass
        
    return ATTRS


def parse_rad_header(file, ATTRS):
    """Parse the R/V Armstrong RAD Met Sensor Header"""
    with open(file) as f:
        header = f.readlines()

    for n, line in enumerate(header):
        # Get the make and model
        if "make" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["shortwave_radiation"].update({key: value})
            ATTRS["longwave_radiation"].update({key:value})
        elif "model" in line.lower():
            line = line.strip() + ' ' + header[n+1].strip()
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["shortwave_radiation"].update({key: value})
            ATTRS["longwave_radiation"].update({key:value})
        elif "calibration date" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["shortwave_radiation"].update({key: value})
            ATTRS["longwave_radiation"].update({key:value})
        elif "s/n" in line.lower():
            if 'swr' in line.lower():
                _, _, sn = line.split()
                sn = sn.strip()
                ATTRS["shortwave_radiation"].update({'serial_number': sn})
            elif 'lwr' in line.lower():
                _, _, sn = line.split()
                sn = sn.strip()
                ATTRS["longwave_radiation"].update({'serial_number': sn})
        elif "installation" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["shortwave_radiation"].update({key: value})
            ATTRS["longwave_radiation"].update({key:value})
        else:
            pass
        
    return ATTRS


def parse_met_header(file, ATTRS):
    """Parse the R/V Armstrong MET sensor header"""
    if 'XTS' in os.path.basename(file):
        attrs = [x for x in ATTRS if 'starboard' in x]
    elif 'XTP' in os.path.basename(file):
        attrs = [x for x in ATTRS if 'port' in x]
    else:
        raise ValueError('File not recognized as port or starboard sensor')
    
    with open(file) as f:
        header = f.readlines()
        
    for line in header:
        if "make" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            for attr in attrs:
                ATTRS[attr].update({key: value})
        elif "model" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            for attr in attrs:
                ATTRS[attr].update({key: value})
        elif "s/n" in line.lower():
            _, sn, _, _, ptu = line.split()
            sn = sn.strip().replace(',','')
            ptu = ptu.strip().replace(',', '')
            for attr in attrs:
                ATTRS[attr].update({'serial_number': sn})
                ATTRS[attr].update({'ptu_serial_number': ptu})
        elif 'calibration date' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            for attr in attrs:
                ATTRS[attr].update({key: value})
        elif 'installation date' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            for attr in attrs:
                ATTRS[attr].update({key: value})
        elif 'installation location' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            for attr in attrs:
                ATTRS[attr].update({key: value})
        elif 'height' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            for attr in attrs:
                ATTRS[attr].update({key: value})
        else:
            pass
    
    return ATTRS


def parse_ssv_header(file, ATTRS):
    """Parse the R/V Armstrong SSV header"""
    with open(file) as f:
        data = f.readlines()
        
    for line in data:
        # Get the make and model
        if "type" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        elif "model" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        elif "calibration date" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        elif "s/n" in line.lower():
            key, value = line.split(':')
            key = 'serial_number'
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        elif 'installation date' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        elif 'installation location' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        elif 'distance' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
        else:
            pass
        
    return ATTRS


def parse_sbe45_header(file, ATTRS):
    """Parse the R/V Armstrong SBE45 thermosalinograph header"""
    with open(file) as f:
        header = f.readlines()
        
    for line in header:
        # Get the make and model
        if "instrument" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["sea_surface_salinity"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        elif "model" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["sea_surface_salinity"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        elif "calibration date" in line.lower():
            key, value = line.split(':')
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        elif "s/n" in line.lower():
            key, value = line.split(':')
            key = 'serial_number'
            value = value.strip()
            ATTRS["sea_surface_salinity"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        elif 'installation date' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["speed_of_sound"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        elif 'installation location' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["sea_surface_salinity"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        elif 'distance' in line.lower():
            key, value = line.split(":")
            key = '_'.join(key.strip().lower().split())
            value = value.strip()
            ATTRS["sea_surface_salinity"].update({key: value})
            ATTRS["sea_surface_temperature"].update({key: value})
        else:
            pass

    return ATTRS


def parse_cruise_metadata(file):
    """Parse the cruise.metadata file for R/V Armstrong"""
    metadata = {}
    
    with open(file) as f:
        header = f.readlines()
        
    notes = False
    while notes == False:
        for n, line in enumerate(header):
            if len(line.strip()) == 0:
                continue
            # Get the make and model
            key, value = line.split(':', 1)
            key = '_'.join(key.strip().lower().split())
            if 'notes' in key.lower():
                notes = True
                value = parse_notes(header[n+1:])
                metadata.update({key: value})
                break
            else:
                value = value.strip()
                metadata.update({key: value})
    
    return metadata

def parse_notes(lines):
    new_lines = ''
    for l in lines:
        if len(l.strip()) == 0:
            continue
        else:
            new_l = " ".join([x.strip() for x in l.split()])
            new_lines = new_lines + ' ' + new_l
    new_lines = new_lines.strip()
    
    return new_lines


def parse_ship_met_data(met_files, ATTRS, met_headers=None, rad_header=None, par_header=None, sbe45_header=None, ssv_header=None, cruise_metadata=None):
    """Parse the ship met files and associated header files."""
    # First, load and parse the met files
    data = parse_data_files(met_files)
    
    # Update the dataset attributes based on headers
    # Now, check for headers and, if they exists, load them
    if met_headers is not None:
        if type(met_headers) is list:
            for hdr in met_headers:
                ATTRS = parse_met_header(hdr, ATTRS)
    if rad_header is not None:
        ATTRS = parse_rad_header(rad_header, ATTRS)
    if par_header is not None:
        ATTRS = parse_par_header(par_header, ATTRS)
    if sbe45_header is not None:
        ATTRS = parse_sbe45_header(sbe45_header, ATTRS)
    if ssv_header is not None:
        ATTRS = parse_ssv_header(ssv_header, ATTRS)
    
    # Now build the dataset
    # Convert the dataframe to a dataset
    ds = xr.Dataset(data)

    # Rename the variables
    for var in ds:
        if var in name_map.keys():
            ds = ds.rename_vars({var: name_map.get(var)})

    # Convert the variables to type float
    for var in ds:
        ds[var] = ds[var].astype(float)

    # Add in the attributes
    for var in ds:
        if var in ATTRS.keys():
            ds[var].attrs = ATTRS.get(var)

    # Add in cruise metadata
    if cruise_metadata is not None:
        metadata = parse_cruise_metadata(cruise_metadata)
        ds.attrs = metadata
        
    # Return the results
    return ds

