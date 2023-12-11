# datasets: motionforecasting 
-----------------------------
- `dataschema`: 
    - **Class** `TrackCategory`: 
        - **Args**: 
            - `TRACK_FRAGMENT`: Low quality track that may only contain a few timestamps of observations.
            - `UNSCORED_TRACK`: Track of reasonable quality, but not scored - can be used for contextual input.
            - `SCORED_TRACK`: High-quality tracks relevant to the AV - scored in the multi-agent prediction challenge.
            - `FOCAL_TRACK`: The track used to generate a particular scenario - scored in the single-agent prediction challenge.
    
    - **class** `ObjectType`: Stores types of detected objects.
        - **Args**: 
            - `VEHICLE` , `PEDESTRIAN` , `MOTORCYCLIST` , `CYCLIST` , `BUS` , `STATIC` , `BACKGROUND` , `CONSTRUCTION` , `RIDERLESS_BICYCLE` , `UNKNOWN`.
    
    - **class** `ObjectState`:Bundles all state information associated with an object at a fixed point in time.
        - **Args**: 
            - `observed`: Boolean indicating if this object state falls in the observed segment of the scenario.
            - `timestep`: Time step corresponding to this object state [0, num_scenario_timesteps).
            - `position`: (x, y) Coordinates of center of object bounding box.
            - `heading`: Heading associated with object bounding box (in radians, defined w.r.t the map coordinate frame).
            - `velocity`: (x, y) Instantaneous velocity associated with the object (in m/s).

    - **class** `Track`: Bundles all data associated with an Argoverse track.
        - **Args**: 
                - `track_id`: Unique ID associated with this track.
                - `object_states`: States for each timestep where the track object had a valid observation.
                - `object_type`: Inferred type for the track object.
                - `category`: Assigned category for track - used as an indicator for prediction requirements and data quality.

    - **class** `ArgoverseScenario`: Bundles all data associated with an Argoverse scenario.
            - `scenario_id`: Unique ID associated with this scenario.
            - `timestamps_ns`: All timestamps associated with this scenario.
            - `tracks`: All tracks associated with this scenario.
            - `focal_track_id`: The track ID associated with the focal agent of the scenario.
            - `city_name`: The name of the city associated with this scenario.
            - `map_id`: The map ID associated with the scenario (used for internal bookkeeping).
            - `slice_id`: ID of the slice used to generate the scenario (used for internal bookkeeping).

------------------------------------------------
- `scenario_serialization`: 
    - **Method** `serialize_argoverse_scenario_parquet`: Serialize a single Argoverse scenario in parquet format and save to disk.
        - **Args**: 
            - `save_path`: Path to save the serialized scenario.
            - `scenario`: ArgoverseScenario object to serialize.
    
    - **Method** `load_argoverse_scenario_parquet`: Load a single Argoverse scenario from disk.
        - **Args**: 
            - `load_path`: Path to load the serialized scenario. path of parquet file.
        - **Returns**:
            - `scenario`: ArgoverseScenario object loaded from disk.

    - **Method** `_convert_tracks_to_tabular_format` converts tracks to tabular format
        - **Args**: 
            - `tracks`: List of tracks to convert to tabular format.
        - **Returns**:
            - DataFrame containing all track data in a tabular format.
    
    - **Method** `_load_tracks_from_tabular_format` Loads tracks from tabular format
        - **Args**: 
            - `tracks_df`: DataFrame containing all track data in a tabular format.
        - **Returns**:
            - List of tracks loaded from tabular format.
------------------------------------------------
- `scenario_visualization`:
    - **Constants:**</br>
        - `_OBS_DURATION_TIMESTEPS`: Number of timesteps to visualize before the focal timestep = `50`.
        - `_PRED_DURATION_TIMESTEPS`: Number of timesteps to visualize after the focal timestep = `60`.
        - `_ESTIMATED_VEHICLE_LENGTH_M`: Estimated length of a vehicle (used for visualization) = `4.0`.
        - `_ESTIMATED_VEHICLE_WIDTH_M`: Estimated width of a vehicle (used for visualization) = `2.0`.
        - `_ESTIMATED_CYCLIST_LENGTH_M`: Estimated length of a cyclist (used for visualization) = `2.0`.
        - `_ESTIMATED_CYCLIST_WIDTH_M`: Estimated width of a cyclist (used for visualization) = `0.7`.
        - `_PLOT_BOUNDS_BUFFER_M`: Buffer (in meters) to add to the plot bounds = `30.0`.
        - `_DERIVABLE_AREA_COLOR`: Color to use for drivable area = `#7A7A7A`.
        - `_FOCAL_AGENT_COLOR`: Color to use for focal agent = `#ECA25B`.
        - `_AV_COLOR`: Color to use for AV = `#007672`.
        - `_BOUNDING_BOX_ZORDER`
        - `_STATIC_OBJECT_TYPES`: Set of object types that are static = `set([ObjectType.STATIC, ObjectType.BACKGROUND, ObjectType.CONSTRUCTION, RIDERLESS_BICYCLE])`.
    
    - **Method:** `visualize_scenario`: Visualize a single Argoverse scenario.
        - **Args**: 
            - `scenario` : ArgoverseScenario object to visualize.
            - `scenario_static_map` : Static map associated with the scenario.
            - `save_path` : Path to save the visualization.
    
    - **Method** `_plot_static_map_elements`: Plot static map elements.
        - **Args**: 
            - `static_map`: Static map associated with the scenario.
            - `show_ped_xings`: Boolean indicating if pedestrian crossings should be plotted.
    
    - **Method** `_plot_actor_tracks`: Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.
        - **Args**: 
            - `ax`: Axes on which actor tracks should be plotted.
            - `scenario`: Argoverse scenario for which to plot actor tracks.
            - `timestep`: Tracks are plotted for all actor data up to the specified time step.
        - **Returns**:
            - `track_bounds`: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.


