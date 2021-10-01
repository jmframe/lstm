#import data_tools
# Basic utilities
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from netCDF4 import Dataset
# Here is the LSTM model we want to run
import nextgen_cuda_lstm
# Configuration file functionality
import yaml
# LSTM here is based on PyTorch
import torch
from torch import nn

class LSTM_run_class():

    def __init__(self):
        """Create a LSTM model driver"""
        super(LSTM_run_class, self).__init__()
        self._values = {}
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        
        self.streamflow_cms = 0.0
        self.streamflow_fms = 0.0
        self.surface_runoff_mm = 0.0

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'LSTM for Next Generation NWM',
        'version':            '1.0',
        'author_name':        'Jonathan Martin Frame',
        'grid_type':          'scalar', # JG Edit
        'time_step_size':      1,       # JG Edit
        'time_units':         '1 hour' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    _input_var_names = [
        'land_surface_radiation~incoming~longwave__energy_flux',
        'land_surface_air__pressure',
        'atmosphere_air_water~vapor__relative_saturation',
        'atmosphere_water__time_integral_of_precipitation_mass_flux',
        'land_surface_radiation~incoming~shortwave__energy_flux',
        'land_surface_air__temperature',
        'land_surface_wind__x_component_of_velocity',
        'land_surface_wind__y_component_of_velocity']

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['land_surface_water__runoff_depth', 
                         'land_surface_water__runoff_volume_flux']

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    # This is going to get long, 
    #     since the input variable names could come from any forcing...
    #------------------------------------------------------
    #_var_name_map_long_first = {
    _var_name_units_map = {
                                'land_surface_water__runoff_volume_flux':['streamflow_cfs','ft3 s-1'],
                                'land_surface_water__runoff_depth':['streamflow_mm','mm'],
                                #--------------   Dynamic inputs --------------------------------
                                'atmosphere_water__time_integral_of_precipitation_mass_flux':['total_precipitation','kg m-2'],
                                'land_surface_radiation~incoming~longwave__energy_flux':['longwave_radiation','W m-2'],
                                'land_surface_radiation~incoming~shortwave__energy_flux':['shortwave_radiation','W m-2'],
                                'atmosphere_air_water~vapor__relative_saturation':['specific_humidity','kg kg-1'],
                                'land_surface_air__pressure':['pressure','Pa'],
                                'land_surface_air__temperature':['temperature','K'],
                                'land_surface_wind__x_component_of_velocity':['wind_u','m s-1'],
                                'land_surface_wind__y_component_of_velocity':['wind_v','m s-1'],
                                #--------------   STATIC Attributes -----------------------------
                                'basin__area':['area_gages2','km2'],
                                'ratio__mean_potential_evapotranspiration__mean_precipitation':['aridity','-'],
                                'basin__carbonate_rocks_area_fraction':['carbonate_rocks_frac','-'],
                                'soil_clay__volume_fraction':['clay_frac','percent'],
                                'basin__mean_of_elevation':['elev_mean','m'],
                                'land_vegetation__forest_area_fraction':['frac_forest','-'],
                                'atmosphere_water__precipitation_falling_as_snow_fraction':['frac_snow','-'],
                                'bedrock__permeability':['geol_permeability','m2'],
                                'land_vegetation__max_monthly_mean_of_green_vegetation_fraction':['gvf_max','-'],
                                'land_vegetation__diff__max_min_monthly_mean_of_green_vegetation_fraction':['gvf_diff','-'],
                                'atmosphere_water__mean_duration_of_high_precipitation_events':['high_prec_dur','d'],
                                'atmosphere_water__frequency_of_high_precipitation_events':['high_prec_freq','d yr-1'],
                                'land_vegetation__diff_max_min_monthly_mean_of_leaf-area_index':['lai_diff','-'],
                                'land_vegetation__max_monthly_mean_of_leaf-area_index':['lai_max','-'],
                                'atmosphere_water__low_precipitation_duration':['low_prec_dur','d'],
                                'atmosphere_water__precipitation_frequency':['low_prec_freq','d yr-1'],
                                'maximum_water_content':['max_water_content','m'],
                                'atmosphere_water__daily_mean_of_liquid_equivalent_precipitation_rate':['p_mean','mm d-1'],
                                'land_surface_water__daily_mean_of_potential_evaporation_flux':['pet_mean','mm d-1'],
                                'basin__mean_of_slope':['slope_mean','m km-1'],
                                'soil__saturated_hydraulic_conductivity':['soil_conductivity','cm hr-1'],
                                'soil_bedrock_top__depth__pelletier':['soil_depth_pelletier','m'],
                                'soil_bedrock_top__depth__statsgo':['soil_depth_statsgo','m'],
                                'soil__porosity':['soil_porosity','-'],
                                'soil_sand__volume_fraction':['sand_frac','percent'],
                                'soil_silt__volume_fraction':['silt_frac','percent'], 
                                'basin_centroid__latitude':['gauge_lat', 'degrees'],
                                'basin_centroid__longitude':['gauge_lon', 'degrees']
                                 }

    #------------------------------------------------------
    # A list of static attributes. Not all these need to be used.
    #------------------------------------------------------
    #   These attributes can be anaything, but usually come from the CAMELS attributes:
    #   Nans Addor Andrew J. Newman, Naoki Mizukami, and Martyn P. Clark
    #   The CAMELS data set: catchment attributes and meteorology for large-sample studies
    #   https://doi.org/10.5194/hess-21-5293-2017
    _static_attributes_list = ['area_gages2','aridity','carbonate_rocks_frac','clay_frac',
                               'elev_mean','frac_forest','frac_snow','geol_permeability',
                               'gvf_max','gvf_diff','high_prec_dur','high_prec_freq','lai_diff',
                               'lai_max','low_prec_dur','low_prec_freq','max_water_content',
                               'p_mean','pet_mean','slope_mean','soil_conductivity',
                               'soil_depth_pelletier','soil_depth_statsgo','soil_porosity',
                               'sand_frac','silt_frac', 'gauge_lat', 'gauge_lon']

    #------------------------------------------------------------
    #------------------------------------------------------------
    # Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def setup_model_for_run( self, driver_cfg_file=None ):

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for long_name in self._var_name_units_map.keys()}
        
        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        for var_name in list(self._var_name_units_map.keys()):
            # ---------- All the variables are single values ------------------#
            # ---------- so just set to zero for now.        ------------------#
            self._values[var_name] = 0
        
        # -------------- Read in the driving configuration -------------------------#
        # This will direct all the next moves.
        if driver_cfg_file is not None:
            with driver_cfg_file.open('r') as fp:
                cfg = yaml.safe_load(fp)
            self.cfg_run = self._parse_config(cfg)
        else:
            print("Error: No configuration provided, nothing to do...")
        
        # ------------- Load in the configuration file for the specific LSTM --#
        # This will include all the details about how the model was trained
        # Inputs, outputs, hyper-parameters, scalers, weights, etc. etc.
        self.get_training_configurations()
        self.get_scaler_values()
        
        # ------------- Initialize an LSTM model ------------------------------#
        self.lstm = nextgen_cuda_lstm.Nextgen_CudaLSTM(input_size=self.input_size, 
                                                       hidden_layer_size=self.hidden_layer_size, 
                                                       output_size=self.output_size, 
                                                       batch_size=1, 
                                                       seq_length=1)

        # ------------ Load in the trained weights ----------------------------#
        # Save the default model weights. We need to make sure we have the same keys.
        default_state_dict = self.lstm.state_dict()

        # Trained model weights from Neuralhydrology.
        trained_model_file = self.cfg_train['run_dir'] / 'model_epoch{}.pt'.format(str(self.cfg_train['epochs']).zfill(3))
        trained_state_dict = torch.load(trained_model_file, map_location=torch.device('cpu'))

        # Changing the name of the head weights, since different in NH
        trained_state_dict['head.weight'] = trained_state_dict.pop('head.net.0.weight')
        trained_state_dict['head.bias'] = trained_state_dict.pop('head.net.0.bias')
        trained_state_dict = {x:trained_state_dict[x] for x in default_state_dict.keys()}

        # Load in the trained weights.
        self.lstm.load_state_dict(trained_state_dict)

        # ------------- Initialize the values for the input to the LSTM  -----#
        self.set_static_attributes()
        self.initialize_forcings()
        
        if self.cfg_run['initial_state'] == 'zero':
            self.h_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
            self.c_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()

        self.t = 0

        # ----------- The output is area normalized, this is needed to un-normalize it
        #                         mm->m                             km2 -> m2          hour->s    
        self.output_factor_cms =  (1/1000) * (self.cfg_run['area_sqkm'] * 1000*1000) * (1/3600)

    #------------------------------------------------------------ 
    def run_single_timestep(self):
        with torch.no_grad():

            self.create_scaled_input_tensor()

            self.lstm_output, self.h_t, self.c_t = self.lstm.forward(self.input_tensor, self.h_t, self.c_t)
            
            self.scale_output()
            
            self.t += 1
    
    
    #------------------------------------------------------------
    #------------------------------------------------------------
    # LSTM: SETUP Functions
    #------------------------------------------------------------
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def get_training_configurations(self):
        if self.cfg_run['train_cfg_file'] is not None:
            with self.cfg_run['train_cfg_file'].open('r') as fp:
                cfg = yaml.safe_load(fp)
            self.cfg_train = self._parse_config(cfg)

        # Collect the LSTM model architecture details from the configuration file
        self.input_size        = len(self.cfg_train['dynamic_inputs']) + len(self.cfg_train['static_attributes'])
        self.hidden_layer_size = self.cfg_train['hidden_size']
        self.output_size       = len(self.cfg_train['target_variables']) 

        # WARNING: This implimentation of the LSTM can only handle a batch size of 1
        self.batch_size        = 1 #self.cfg_train['batch_size']

        # Including a list of the model input names.
        self.all_lstm_inputs = []
        self.all_lstm_inputs.extend(self.cfg_train['dynamic_inputs'])
        self.all_lstm_inputs.extend(self.cfg_train['static_attributes'])
        
        # Scaler data from the training set. This is used to normalize the data (input and output).
        with open(self.cfg_train['run_dir'] / 'train_data' / 'train_data_scaler.p', 'rb') as fb:
            self.train_data_scaler = pickle.load(fb)

    #------------------------------------------------------------ 
    def get_scaler_values(self):

        """Mean and standard deviation for the inputs and LSTM outputs""" 

        self.out_mean = self.train_data_scaler['xarray_feature_center'][self.cfg_train['target_variables'][0]].values
        self.out_std = self.train_data_scaler['xarray_feature_scale'][self.cfg_train['target_variables'][0]].values

        self.input_mean = []
        self.input_mean.extend([self.train_data_scaler['xarray_feature_center'][x].values for x in self.cfg_train['dynamic_inputs']])
        self.input_mean.extend([self.train_data_scaler['attribute_means'][x] for x in self.cfg_train['static_attributes']])
        self.input_mean = np.array(self.input_mean)

        self.input_std = []
        self.input_std.extend([self.train_data_scaler['xarray_feature_scale'][x].values for x in self.cfg_train['dynamic_inputs']])
        self.input_std.extend([self.train_data_scaler['attribute_stds'][x] for x in self.cfg_train['static_attributes']]) 
        self.input_std = np.array(self.input_std)

    #------------------------------------------------------------ 
    def create_scaled_input_tensor(self):
        
        self.input_array = np.array([self._values[self._var_name_map_short_first[x]] for x in self.all_lstm_inputs])
        
        self.input_array_scaled = (self.input_array - self.input_mean) / self.input_std 
        self.input_tensor = torch.tensor(self.input_array_scaled)
        
    #------------------------------------------------------------ 
    def scale_output(self):

        if self.cfg_train['target_variables'][0] == 'qobs_mm_per_hour':
            self.surface_runoff_mm = (self.lstm_output[0,0,0].numpy().tolist() * self.out_std + self.out_mean)

        elif self.cfg_train['target_variables'][0] == 'QObs(mm/d)':
            self.surface_runoff_mm = (self.lstm_output[0,0,0].numpy().tolist() * self.out_std + self.out_mean) * (1/24)
            
        self._values['land_surface_water__runoff_depth'] = self.surface_runoff_mm
        self.streamflow_cms = self.surface_runoff_mm * self.output_factor_cms

        self._values['land_surface_water__runoff_volume_flux'] = self.streamflow_cms * (1/35.314)

    #-------------------------------------------------------------------
    def read_initial_states(self):
        h_t = np.genfromtxt(self.h_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.h_t = torch.tensor(h_t).view(1,1,-1)
        c_t = np.genfromtxt(self.c_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.c_t = torch.tensor(c_t).view(1,1,-1)

    #---------------------------------------------------------------------------- 
    def set_static_attributes(self):
        """ Get the static attributes from the configuration file
        """
        for attribute in self._static_attributes_list:
            if attribute in self.cfg_train['static_attributes']:
                
                long_var_name = self._var_name_map_short_first[attribute]

                # This is probably the better way to do it,
                setattr(self, long_var_name, self.cfg_run[attribute])
                
                # and this is just in case. _values dictionary is in the example
                self._values[long_var_name] = self.cfg_run[attribute]
    
    #---------------------------------------------------------------------------- 
    def initialize_forcings(self):
        for forcing_name in self.cfg_train['dynamic_inputs']:
            setattr(self, self._var_name_map_short_first[forcing_name], 0)

    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #-- Random utility functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 

    def _parse_config(self, cfg):
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Add more config parsing if necessary
        return cfg


# creating an instance of an LSTM model
print('creating an instance of an LSTM object')
model = LSTM_run_class()

# Initializing the BMI
print('Setting up the model to run')
model.setup_model_for_run(driver_cfg_file=Path('./run_config_files/01022500_hourly_all_attributes_forcings.yml'))

# Get input data that matches the LSTM test runs
print('Get input data that matches the LSTM test runs')
sample_data = Dataset(Path('./data/usgs-streamflow-nldas_hourly.nc'), 'r')

# Now loop through the inputs, set the forcing values, and update the model
print('Now loop through the inputs, set the forcing values, and update the model')
for precip, temp in zip(list(sample_data['total_precipitation'][3].data),
                        list(sample_data['temperature'][3].data)):

    model._values['atmosphere_water__time_integral_of_precipitation_mass_flux'] = precip
    model._values['land_surface_air__temperature'] = temp

    model.run_single_timestep()

    print('the streamflow (CFS) at time {} is {:.2f}'.format(model.t, model._values['land_surface_water__runoff_volume_flux']))

    if model.t > 2*365*24:
        print('stopping the loop')
        break