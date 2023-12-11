Geometry: 
---------
- `Camera`: 
	- `pihole_camera`: </br>
		**Class** `intrinsics`: Stores intrinsic matrix of camera in (K).</br>
		**Class** `PinholeCamera`: </br>
	    - Args: </br>
				`intrinsics`: Intrinsic object Intrinsic matrix of camera.</br>
				`extrinsics`: SE3 Object Extrinsic matrix of camera.</br>
				`cam_name`: Name of camera.</br>
        - Methods:</br> 
				- **method**  `Getters` for all the above attributes.</br>
				- **method**  `from_feather`: Takes feather file path that contains camera data and returns PinholeCamera object.</br>
				- **method**  `cull_to_view_frustum`: Given a set of coordinates in the image plane and corresponding points in the camera coordinate reference frame, points that have a valid projection into the image. 3d points with valid projections have x coordinates in the range [0,width_px-1], y-coordinates in the range [0,height_px-1], and a positive z-coordinate (lying in front of the camera frustum).</br>
				- **method**  `project_ego_to_img`: Project a collection of 3d points (provided in the egovehicle frame) to the image plane.</br>
				- **method**  `project_cam_to_img`: Project a collection of 3d points in the camera reference frame to the image plane.</br>
				- **method**  `project_ego_to_img_motion_compensated`: Project points in the ego frame to the image with motion compensation.</br> 
				- **method**  `right_clipping_plane`:</br>
				- **method**  `left_clipping_plane`: </br>
				- **method**  `top_clipping_plane`: </br>
				- **method**  `bottom_clipping_plane`: </br>
				- **method**  `near_clipping_plane`: </br>
				- **method**  `frustum_planes`: compute all 5 previous planes</br>
				- **method**  `egovehicle_yaw_cam_rad`: Returns Counter-clockwise angle from x=0 (in radians) of camera center ray, in the egovehicle frame.</br>
				- **method**  `fov_theta_rad`: Compute the field of view of a camera frustum to use for view frustum culling during rendering.</br>
				- **method**  `compute_pixel_ray_directions`: Given (u,v) coordinates and intrinsics, generate pixel rays in the camera coordinate frame.</br>

        
		**Method** `remove_nan_values`: Takes two matrices -> removes rows that have null values in first matrix from both matrices.</br>

- `geometry`:</br>
	- `Methods`: </br>
		- `wrap_angles`: Map angles (in radians) from domain [-∞, ∞] to [0, π).</br>
		- **method**  `xy_to_uv`: Convert coordinates in R^2 (x,y) to texture coordinates (u,v) in R^2.</br>
		- **method**  `quat_to_mat`: Convert a quaternion to a 3D rotation matrix. </br>
		- **method**  `mat_to_quat`: Convert a 3D rotation matrix to a quaternion.</br>
		- **method**  `mat_to_xyz` : Convert a 3D rotation matrix to a sequence of _extrinsic_ rotations.</br>
		- **method**  `xyz_to_mat` : Convert a sequence of rotations about the (x,y,z) axes to a 3D rotation matrix.</br>
		- **method**  `cart_to_sph`: Convert Cartesian coordinates into spherical coordinates.</br>
		- **method**  `cart_to_hom`: Convert Cartesian coordinates to homogeneous coordinates.</br>
		- **method**  `hom_to_cart`: Convert homogeneous coordinates to Cartesian coordinates.</br>
		- **method**  `crop_points`: takes group of points and crops them to be between lower and upper bounds.</br>
		- **method**  `compute_interior_points_mask`:</br> 
		<font style="color:cyan	;font-weight:200">Args:</font>
        **points_xyz**: (N,3) Array representing a point cloud in Cartesian coordinates (x,y,z).</br>
        **cuboid_vertices**: (8,3) Array representing 3D cuboid vertices, ordered as shown above. </br>
    <font style="color:Green;font-weight:200">Returns: </font>
        (N,) An array of boolean flags indicating whether the points are interior to the cuboid.</br	>

- `iou`: </br>
	- **method**  `iou_3d_axis_aligned`: Compute 3d, axis-aligned (vertical axis alignment) intersection-over-union (IoU) between two sets of cuboids.</br>

- `mesh_grid`: </br>
	- **method**  `get_mesh_grid_as_point_cloud`: returns mesh grid of x values between min and max x and y values between min and max y.</br>

- `poly_line_utils`: </br>
	- **method**  `get_polyline_length`: Returns length of polyline.</br>
	- **method**  `interp_polyline_by_fixed_waypt_interval`: Resample waypoints of a polyline so that waypoints appear roughly at fixed intervals from the start.</br>
	- **method**  `get_double_polylines`: Treat any polyline as a centerline, and extend a narrow strip on both sides.</br>
	- **method**  `swap_left_and_right`: Swap points in left and right centerline according to condition.</br>
	- **method**  `convert_lane_boundaries_to_polygon`: Convert lane boundaries to polygon.</br>

- `se3`: </br>
	- **class** `SE3`: </br>
		- **Args**: </br>
			`rotation`: Rotation matrix.</br>
			`translation`: Translation vector.</br>
		- **Methods**: </br>
			- **method** `transform_matrix`: Takes rotation and translation matrix and returns SE3 object.</br>
			- **method** `inverse`: Returns inverse of SE3 object.</br>
			- **method** `compose`: Compose two SE3 objects.</br>
			- **method** `transform_point_cloud`: Transform a point cloud by an SE3 object.</br>

- `sim2`
	- **class** `Sim2`: </br>
		- **Args**: </br>
			`rotation`: Rotation matrix.</br>
			`translation`: Translation vector.</br>
			`scale`: Scale factor.</br>
		- **Methods**: </br>
			- **method** `transform_matrix`: Takes rotation, translation and scale matrix and returns Sim2 object.</br>
			- **method** `inverse`: Returns inverse of Sim2 object.</br>
			- **method** `compose`: Compose two Sim2 objects.</br>
			- **method** `transform_point_cloud`: Transform a point cloud by an Sim2 object.</br>
			- **method** `save_as_json`: Save Sim2 object as json file.</br>
			- **method** `from_json`: Load Sim2 object from json file.</br>
			- **method** `from_matrix`: Load Sim2 object from matrix.</br>

- `utm`: 
	- **class** `CityName`: stores city names, All are North UTM zones (Northern hemisphere), city longitude and latitude.</br>
	- **method** `convert_gps_to_utm`: Convert GPS coordinates to UTM coordinates.</br>	
		- Args: </br>
			`lat`: Latitude.</br>
			`lon`: Longitude.</br>
			`city_name`: City name.</br>
		- Returns: </br>
			`UTM coordinates`.</br>
	- **method** `convert_city_coords_to_utm`: Convert city coordinates to UTM coordinates.</br>
		- Args: </br>
			`points_city`: (N,2) array, representing 2d query points in the city coordinate frame.</br>
			`city_name`: City name.</br>
		- Returns: </br>
			`points_utm`: Array of shape (N,2), representing points in the UTM coordinate system, as (easting, northing)..</br>
	- **method** `convert_city_coords_to_wgs84`: Convert city coordinates to WGS84 coordinates.</br>
		- Args: </br>
			`points_city`: (N,2) array, representing 2d query points in the city coordinate frame.</br>
			`city_name`: City name.</br>
		- Returns: </br>
			`points_wgs84`: Array of shape (N,2), representing points in the WGS84 coordinate system, as (latitude, longitude).</br>