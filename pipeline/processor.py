"""Logic for HAND calculation, Cloud Masking, & Alignment."""

import logging
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from pysheds.grid import Grid
from scipy.interpolate import interp1d
from typing import List

logger = logging.getLogger(__name__)

def calculate_hand(dem_data, acc_threshold=100):
    """
    Calculates Height Above Nearest Drainage (HAND) from DEM data.
    
    Args:
        dem_data: xarray DataArray with DEM values and geospatial info, or path to DEM file
        acc_threshold: Accumulation threshold for stream initiation (default: 100)
    
    Returns:
        xarray DataArray with HAND values and geospatial info
    """
    # Handle both file paths and xarray DataArrays
    if isinstance(dem_data, str):
        # If it's a file path, read it
        dem_xr = rioxarray.open_rasterio(dem_data)
        # Use np.asarray(...) instead of deprecated .values for compatibility
        dem_array = np.asarray(dem_xr).squeeze()
        dem_xr_template = dem_xr
    elif isinstance(dem_data, xr.DataArray):
        # If it's already an xarray DataArray
        dem_array = np.asarray(dem_data).squeeze()
        dem_xr_template = dem_data
    else:
        raise ValueError("dem_data must be either a file path (str) or xarray DataArray")

    # Hard fail early on empty DEM crops (prevents obscure downstream numpy reduction errors)
    if dem_array.size == 0:
        raise ValueError("DEM array is empty after loading/clipping. Check bbox and CRS handling.")
    
    # Create temporary file for pysheds (it needs a file path)
    import tempfile
    import os
    from pathlib import Path
    
    # Save to temporary file
    temp_dir = Path(tempfile.gettempdir())
    temp_dem_path = temp_dir / "temp_dem_for_hand.tif"
    
    # Ensure we have a 2D array
    if len(dem_array.shape) == 3:
        dem_array = dem_array.squeeze(axis=0)
    
    # Create a temporary xarray with the DEM data
    temp_dem_xr = xr.DataArray(
        dem_array,
        dims=['y', 'x'],
        coords={
            'y': np.arange(dem_array.shape[0]),
            'x': np.arange(dem_array.shape[1])
        }
    )
    
    # Copy geospatial info if available
    if hasattr(dem_xr_template, 'rio'):
        if hasattr(dem_xr_template.rio, 'crs') and dem_xr_template.rio.crs is not None:
            temp_dem_xr.rio.write_crs(dem_xr_template.rio.crs, inplace=True)
        if hasattr(dem_xr_template.rio, 'transform'):
            transform = dem_xr_template.rio.transform()
            if transform is not None:
                temp_dem_xr.rio.write_transform(transform, inplace=True)
    
    # Save to temporary file
    temp_dem_xr.rio.to_raster(str(temp_dem_path))
    
    try:
        # Initialize pysheds Grid
        grid = Grid.from_raster(str(temp_dem_path))
        dem_grid = grid.read_raster(str(temp_dem_path))
        
        # Fill depressions in DEM
        filled_dem = grid.fill_depressions(dem_grid)
        
        # Calculate flow direction
        flow_dir = grid.resolve_flats(filled_dem)
        
        # Calculate flow accumulation
        accumulation = grid.accumulation(flow_dir)
        
        # Extract drainage network (where accumulation exceeds threshold)
        streams = accumulation > acc_threshold
        
        # Calculate HAND: distance from each pixel to nearest stream
        # This is a simplified version - full HAND calculation is more complex
        # For now, we'll use a distance transform approach
        from scipy.ndimage import distance_transform_edt
        
        # Create binary mask of streams
        stream_mask = streams.astype(float)
        
        # Calculate distance to nearest stream (inverse of HAND concept)
        # HAND = elevation - elevation of nearest drainage point
        # Simplified: use distance to stream network as proxy
        distance_to_stream = distance_transform_edt(1 - stream_mask)
        
        # HAND is roughly proportional to distance from streams
        # Normalize and scale appropriately
        hand_array = distance_to_stream.astype(np.float32)
        
        # Create output xarray with same geospatial info as input
        # Only use y and x coords, exclude band dimension
        coords_dict = {}
        if hasattr(dem_xr_template, 'coords'):
            for dim in ['y', 'x']:
                if dim in dem_xr_template.coords:
                    coords_dict[dim] = dem_xr_template.coords[dim]
        
        hand_xr = xr.DataArray(
            hand_array,
            dims=['y', 'x'],
            coords=coords_dict if coords_dict else None
        )
        
        # Copy geospatial attributes
        if hasattr(dem_xr_template, 'rio'):
            if hasattr(dem_xr_template.rio, 'crs') and dem_xr_template.rio.crs is not None:
                hand_xr.rio.write_crs(dem_xr_template.rio.crs, inplace=True)
            if hasattr(dem_xr_template.rio, 'transform'):
                transform = dem_xr_template.rio.transform()
                if transform is not None:
                    hand_xr.rio.write_transform(transform, inplace=True)
        
        return hand_xr
        
    except Exception as e:
        # Fallback: Use simplified HAND based on DEM elevation gradients
        # This avoids pysheds/numba compatibility issues
        logger.warning(f"HAND calculation failed with pysheds: {e}")
        logger.warning("Using simplified HAND based on DEM gradient instead")
        
        from scipy.ndimage import gaussian_filter, sobel
        
        # Smooth DEM to reduce noise
        smoothed_dem = gaussian_filter(dem_array, sigma=2.0)
        
        # Calculate gradient magnitude (slope)
        grad_y = sobel(smoothed_dem, axis=0)
        grad_x = sobel(smoothed_dem, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # HAND is inversely related to gradient (steeper = lower HAND, flatter = higher HAND)
        # Normalize to 0-1 range
        # Check for empty arrays first
        if gradient_magnitude.size == 0:
            logger.warning("Empty gradient_magnitude array, using constant HAND")
            hand_array = np.ones_like(dem_array, dtype=np.float32) * 0.5
        elif gradient_magnitude.max() > gradient_magnitude.min():
            hand_array = 1.0 - ((gradient_magnitude - gradient_magnitude.min()) / 
                               (gradient_magnitude.max() - gradient_magnitude.min()))
        else:
            # If no gradient variation, use constant
            hand_array = np.ones_like(dem_array, dtype=np.float32) * 0.5
        
        hand_array = hand_array.astype(np.float32)
        
        # Create output xarray with same geospatial info as input
        # Only use y and x coords, exclude band dimension
        coords_dict = {}
        if hasattr(dem_xr_template, 'coords'):
            for dim in ['y', 'x']:
                if dim in dem_xr_template.coords:
                    coords_dict[dim] = dem_xr_template.coords[dim]
        
        hand_xr = xr.DataArray(
            hand_array,
            dims=['y', 'x'],
            coords=coords_dict if coords_dict else None
        )
        
        # Copy geospatial attributes
        if hasattr(dem_xr_template, 'rio'):
            if hasattr(dem_xr_template.rio, 'crs') and dem_xr_template.rio.crs is not None:
                hand_xr.rio.write_crs(dem_xr_template.rio.crs, inplace=True)
            if hasattr(dem_xr_template.rio, 'transform'):
                transform = dem_xr_template.rio.transform()
                if transform is not None:
                    hand_xr.rio.write_transform(transform, inplace=True)
        
        return hand_xr
        
    finally:
        # Clean up temporary file
        if temp_dem_path.exists():
            try:
                os.remove(temp_dem_path)
            except:
                pass

def mask_clouds(modis_stack: xr.DataArray, qa_band: xr.DataArray) -> xr.DataArray:
    """
    Uses MODIS QA bits to mask clouds and interpolates values temporally.
    Bits 0-1 of MOD09A1 QA: 0=clear, 1=cloudy, 2=mixed, 3=not set

    Args:
        modis_stack: xarray DataArray with shape (time, bands, y, x)
        qa_band: xarray DataArray with QA values (time, y, x)

    Returns:
        xarray DataArray with cloud-masked and interpolated values
    """
    # Sort by time FIRST to ensure monotonic increasing (required for interpolation)
    if 'time' in modis_stack.dims:
        modis_stack = modis_stack.sortby('time')
    if 'time' in qa_band.dims:
        qa_band = qa_band.sortby('time')
    
    # Create cloud mask (bits 0-1: 0=clear, anything else is cloud/shadow)
    # Convert to integer if needed for bitwise operations
    qa_values = np.asarray(qa_band)
    if not np.issubdtype(qa_values.dtype, np.integer):
        qa_values = qa_values.astype(np.uint16)
    cloud_mask = np.bitwise_and(qa_values, 3) != 0
    
    # Convert to xarray for easier handling
    if not isinstance(cloud_mask, xr.DataArray):
        cloud_mask = xr.DataArray(cloud_mask, dims=qa_band.dims, coords=qa_band.coords)
    
    # Apply mask to MODIS stack
    masked_stack = modis_stack.where(~cloud_mask)
    
    # Temporal interpolation for each pixel and band
    # Use linear interpolation along time dimension
    try:
        interpolated = masked_stack.interpolate_na(dim='time', method='linear')
    except ValueError as e:
        # If interpolation fails, try forward/backward fill instead
        # This can happen frequently due to duplicate/non-monotonic time stamps in MODIS stacks.
        # It's not fatal; forward/backward fill is an acceptable fallback, so keep log noise low.
        logger.info(f"Linear interpolation failed: {e}. Using forward/backward fill.")
        interpolated = masked_stack
    
    # Fill any remaining NaN values with forward/backward fill
    interpolated = interpolated.ffill(dim='time').bfill(dim='time')
    
    return interpolated

def align_to_reference(data, reference, resampling='bilinear'):
    """
    Resamples data to match reference's CRS, extent, and resolution.
    
    Args:
        data: xarray DataArray to resample
        reference: xarray DataArray with target CRS/extent/resolution
        resampling: Resampling method string ('bilinear', 'nearest', 'cubic', etc.)
                    or rasterio Resampling enum
    
    Returns:
        Resampled xarray DataArray aligned to reference
    """
    # Convert string to rasterio Resampling enum if needed
    if isinstance(resampling, str):
        resampling_map = {
            'nearest': rasterio.enums.Resampling.nearest,
            'bilinear': rasterio.enums.Resampling.bilinear,
            'cubic': rasterio.enums.Resampling.cubic,
            'cubic_spline': rasterio.enums.Resampling.cubic_spline,
            'lanczos': rasterio.enums.Resampling.lanczos,
            'average': rasterio.enums.Resampling.average,
            'mode': rasterio.enums.Resampling.mode,
            'max': rasterio.enums.Resampling.max,
            'min': rasterio.enums.Resampling.min,
            'med': rasterio.enums.Resampling.med,
            'q1': rasterio.enums.Resampling.q1,
            'q3': rasterio.enums.Resampling.q3,
            'sum': rasterio.enums.Resampling.sum,
            'rms': rasterio.enums.Resampling.rms
        }
        resampling = resampling_map.get(resampling.lower(), rasterio.enums.Resampling.bilinear)
    
    try:
        # Try reproject_match first (works when datasets overlap)
        aligned = data.rio.reproject_match(reference, resampling=resampling)
        return aligned
    except (ValueError, KeyError) as e:
        # If reproject_match fails (e.g., due to dimension mismatches),
        # use explicit reproject with reference's parameters
        logger.warning(f"reproject_match failed: {e}. Using explicit reproject instead.")
        
        # Get target CRS, transform, and shape from reference
        if not hasattr(reference, 'rio') or not hasattr(reference.rio, 'crs'):
            raise ValueError("Reference raster must have rio CRS attributes")
        
        target_crs = reference.rio.crs
        if target_crs is None:
            raise ValueError("Reference raster CRS is None. Cannot reproject without target CRS.")
        
        target_transform = reference.rio.transform()
        
        # Get output shape from reference (handle 2D, 3D, 4D)
        if len(reference.shape) >= 2:
            target_height = reference.shape[-2]
            target_width = reference.shape[-1]
        else:
            target_height, target_width = reference.shape
        
        # Reproject with explicit target parameters
        aligned = data.rio.reproject(
            dst_crs=target_crs,
            transform=target_transform,
            shape=(target_height, target_width),
            resampling=resampling
        )
        
        return aligned

def validate_and_align_arrays(reference_array, *arrays_to_align, method='crop'):
    """
    Ensures all arrays have exactly the same spatial dimensions and coordinates.
    
    This function validates that all input arrays match the reference array's
    spatial dimensions (height, width) and coordinates. If they don't match,
    it crops or pads them to match exactly.
    
    Args:
        reference_array: xarray DataArray with target dimensions and coords
        *arrays_to_align: Variable number of xarray DataArrays to align
        method: 'crop' (default) or 'pad'. If 'crop', crops larger arrays to match.
                If 'pad', pads smaller arrays to match. 'crop' is safer for training.
    
    Returns:
        tuple: (reference_array, aligned_array1, aligned_array2, ...)
               All arrays will have matching spatial dimensions and coordinates.
    """
    logger.info("Validating and aligning arrays to match reference dimensions...")
    
    # Get reference spatial dimensions and coordinates
    if len(reference_array.shape) < 2:
        raise ValueError(f"Reference array must have at least 2 dimensions, got {len(reference_array.shape)}")
    
    ref_height = reference_array.shape[-2]
    ref_width = reference_array.shape[-1]
    ref_y_coords = reference_array.coords.get('y', None)
    ref_x_coords = reference_array.coords.get('x', None)
    
    # If coordinates don't exist, try to create them from dims
    if ref_y_coords is None and 'y' in reference_array.dims:
        ref_y_coords = reference_array.coords['y']
    if ref_x_coords is None and 'x' in reference_array.dims:
        ref_x_coords = reference_array.coords['x']
    
    logger.debug(f"Reference dimensions: {ref_height} x {ref_width}")
    
    aligned_arrays = [reference_array]
    
    for i, arr in enumerate(arrays_to_align):
        if arr is None:
            aligned_arrays.append(None)
            continue
            
        # Get array spatial dimensions
        if len(arr.shape) < 2:
            raise ValueError(f"Array {i+1} must have at least 2 dimensions, got {len(arr.shape)}")
        arr_height = arr.shape[-2]
        arr_width = arr.shape[-1]
        
        # Check if dimensions match
        if arr_height == ref_height and arr_width == ref_width:
            # Dimensions match, but verify coordinates match
            arr_y_coords = arr.coords.get('y', None)
            arr_x_coords = arr.coords.get('x', None)
            
            if (ref_y_coords is not None and arr_y_coords is not None and
                len(ref_y_coords) == len(arr_y_coords) and
                np.allclose(np.asarray(ref_y_coords), np.asarray(arr_y_coords), rtol=1e-6)):
                if (ref_x_coords is not None and arr_x_coords is not None and
                    len(ref_x_coords) == len(arr_x_coords) and
                    np.allclose(np.asarray(ref_x_coords), np.asarray(arr_x_coords), rtol=1e-6)):
                    # Everything matches, use as-is
                    aligned_arrays.append(arr)
                    logger.debug(f"Array {i+1}: dimensions match ({arr_height}x{arr_width})")
                    continue
            
            # Coordinates don't match, reindex to reference coordinates
            logger.warning(f"Array {i+1}: dimensions match but coordinates differ, reindexing...")
            try:
                arr_aligned = arr.reindex(y=ref_y_coords, x=ref_x_coords, method='nearest')
                aligned_arrays.append(arr_aligned)
                continue
            except Exception as e:
                logger.warning(f"Reindexing failed: {e}. Will crop/pad instead.")
        
        # Dimensions don't match - need to crop or pad
        logger.warning(f"Array {i+1}: dimensions mismatch ({arr_height}x{arr_width} vs {ref_height}x{ref_width}), {method}ing...")
        
        if method == 'crop':
            # Crop larger arrays to match reference
            if arr_height > ref_height or arr_width > ref_width:
                # Calculate crop indices (center crop)
                y_start = (arr_height - ref_height) // 2
                y_end = y_start + ref_height
                x_start = (arr_width - ref_width) // 2
                x_end = x_start + ref_width
                
                # Handle different array dimensions (2D, 3D, 4D)
                if len(arr.shape) == 2:
                    arr_cropped = arr.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                elif len(arr.shape) == 3:
                    arr_cropped = arr.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                elif len(arr.shape) == 4:
                    arr_cropped = arr.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                else:
                    # For other dimensions, use isel on last two dims
                    slices = [slice(None)] * (len(arr.shape) - 2) + [slice(y_start, y_end), slice(x_start, x_end)]
                    arr_cropped = arr.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                
                # Reindex to match reference coordinates exactly
                if ref_y_coords is not None and ref_x_coords is not None:
                    arr_cropped = arr_cropped.reindex(y=ref_y_coords, x=ref_x_coords, method='nearest')
                
                aligned_arrays.append(arr_cropped)
            else:
                # Array is smaller - pad with NaN/zeros
                logger.warning(f"Array {i+1} is smaller than reference, padding with zeros...")
                # Create new array with reference dimensions
                new_shape = list(arr.shape)
                new_shape[-2] = ref_height
                new_shape[-1] = ref_width
                
                # Build padded numpy array first, then wrap in DataArray
                arr_np = np.asarray(arr)
                padded_np = np.zeros(new_shape, dtype=arr_np.dtype)
                y_offset = (ref_height - arr_height) // 2
                x_offset = (ref_width - arr_width) // 2
                
                if len(arr.shape) == 2:
                    padded_np[y_offset:y_offset+arr_height, x_offset:x_offset+arr_width] = arr_np
                elif len(arr.shape) == 3:
                    padded_np[:, y_offset:y_offset+arr_height, x_offset:x_offset+arr_width] = arr_np
                elif len(arr.shape) == 4:
                    padded_np[:, :, y_offset:y_offset+arr_height, x_offset:x_offset+arr_width] = arr_np
                
                arr_padded = xr.DataArray(
                    padded_np,
                    dims=arr.dims,
                    coords={**arr.coords, 'y': ref_y_coords, 'x': ref_x_coords} if ref_y_coords is not None else arr.coords
                )
                
                aligned_arrays.append(arr_padded)
        else:
            # Pad method - always pad smaller arrays
            logger.warning(f"Padding array {i+1} to match reference...")
            new_shape = list(arr.shape)
            new_shape[-2] = ref_height
            new_shape[-1] = ref_width
            
            # Build padded numpy array first, then wrap in DataArray
            arr_np = np.asarray(arr)
            padded_np = np.zeros(new_shape, dtype=arr_np.dtype)
            y_offset = (ref_height - arr_height) // 2
            x_offset = (ref_width - arr_width) // 2
            
            if len(arr.shape) == 2:
                padded_np[y_offset:y_offset+arr_height, x_offset:x_offset+arr_width] = arr_np
            elif len(arr.shape) == 3:
                padded_np[:, y_offset:y_offset+arr_height, x_offset:x_offset+arr_width] = arr_np
            elif len(arr.shape) == 4:
                padded_np[:, :, y_offset:y_offset+arr_height, x_offset:x_offset+arr_width] = arr_np
            
            arr_padded = xr.DataArray(
                padded_np,
                dims=arr.dims,
                coords={**arr.coords, 'y': ref_y_coords, 'x': ref_x_coords} if ref_y_coords is not None else arr.coords
            )
            
            aligned_arrays.append(arr_padded)
    
    # Final validation - ensure all arrays have matching dimensions
    for i, arr in enumerate(aligned_arrays):
        if arr is not None:
            assert arr.shape[-2] == ref_height and arr.shape[-1] == ref_width, \
                f"Array {i} still has wrong dimensions: {arr.shape[-2:]} vs {ref_height}x{ref_width}"
    
    logger.info(f"All arrays validated and aligned to {ref_height}x{ref_width}")
    return tuple(aligned_arrays)

def safe_reindex_to_reference(data_array, reference_array):
    """
    Safely reindex a data array to match a reference array's coordinates.
    
    Args:
        data_array: xarray DataArray to reindex
        reference_array: xarray DataArray with target coordinates
    
    Returns:
        Reindexed DataArray matching reference coordinates
    """
    try:
        # Check if both arrays have y and x coordinates
        if 'y' in reference_array.coords and 'x' in reference_array.coords:
            if 'y' in data_array.dims and 'x' in data_array.dims:
                return data_array.reindex(y=reference_array.coords['y'], 
                                         x=reference_array.coords['x'], 
                                         method='nearest')
        # If reindexing fails or coords don't exist, return original
        logger.warning("Could not reindex data_array to reference coordinates, using original")
        return data_array
    except Exception as e:
        logger.warning(f"Reindexing failed: {e}, using original array")
        return data_array

def create_patches(data_array, patch_size=64, stride=32):
    """
    Slices a large array into smaller overlapping patches.
    
    Args:
        data_array: Input array of shape:
                   - 2D: (height, width) - for labels
                   - 3D: (channels, height, width) - for static features
                   - 4D: (time, channels, height, width) - for time series
        patch_size: Height/width of each square patch
        stride: How many pixels to move for the next patch (stride < patch_size creates overlap)
        
    Returns:
        numpy array of patches: (N, ...patch_dims)
    """
    ndim = len(data_array.shape)
    
    if ndim == 2:
        # 2D: (h, w) - labels
        h, w = data_array.shape
        patches = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = data_array[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        return np.array(patches)
    
    elif ndim == 3:
        # 3D: (c, h, w) - static features
        c, h, w = data_array.shape
        patches = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = data_array[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        return np.array(patches)
    
    elif ndim == 4:
        # 4D: (time, c, h, w) - time series
        t, c, h, w = data_array.shape
        patches = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = data_array[:, :, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        return np.array(patches)
    else:
        raise ValueError(f"Unsupported array dimension: {ndim}. Expected 2D, 3D, or 4D.")

def reconstruct_from_patches(patches, original_shape, patch_size, stride):
    """
    Reconstructs a full array from patches.
    
    Args:
        patches: Array of patches (N, ...patch_dims)
        original_shape: Target shape of reconstructed array (h, w) or (c, h, w) or (t, c, h, w)
        patch_size: Size of patches used
        stride: Stride used when creating patches
        
    Returns:
        Reconstructed array with original_shape
    """
    ndim = len(original_shape)
    patches = np.array(patches)
    
    if ndim == 2:
        # 2D: (h, w)
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=patches.dtype)
        count = np.zeros((h, w), dtype=np.int32)
        
        patch_idx = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = patches[patch_idx]
                reconstructed[y:y+patch_size, x:x+patch_size] += patch
                count[y:y+patch_size, x:x+patch_size] += 1
                patch_idx += 1
        
        # Average overlapping regions
        reconstructed = reconstructed / np.maximum(count, 1)
        return reconstructed
    
    elif ndim == 3:
        # 3D: (c, h, w)
        c, h, w = original_shape
        reconstructed = np.zeros((c, h, w), dtype=patches.dtype)
        count = np.zeros((h, w), dtype=np.int32)
        
        patch_idx = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = patches[patch_idx]
                reconstructed[:, y:y+patch_size, x:x+patch_size] += patch
                count[y:y+patch_size, x:x+patch_size] += 1
                patch_idx += 1
        
        # Average overlapping regions
        for i in range(c):
            reconstructed[i] = reconstructed[i] / np.maximum(count, 1)
        return reconstructed
    
    elif ndim == 4:
        # 4D: (t, c, h, w)
        t, c, h, w = original_shape
        reconstructed = np.zeros((t, c, h, w), dtype=patches.dtype)
        count = np.zeros((h, w), dtype=np.int32)
        
        patch_idx = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = patches[patch_idx]
                reconstructed[:, :, y:y+patch_size, x:x+patch_size] += patch
                count[y:y+patch_size, x:x+patch_size] += 1
                patch_idx += 1
        
        # Average overlapping regions
        for i in range(t):
            for j in range(c):
                reconstructed[i, j] = reconstructed[i, j] / np.maximum(count, 1)
        return reconstructed
    else:
        raise ValueError(f"Unsupported original_shape dimension: {ndim}")

def reconstruct_from_patches(patches, original_shape, patch_size, stride):
    """
    Reconstructs a full array from patches.
    
    Args:
        patches: Array of patches (N, ...patch_dims)
        original_shape: Target shape of reconstructed array (h, w) for 2D output
        patch_size: Size of patches used
        stride: Stride used when creating patches
        
    Returns:
        Reconstructed array with original_shape
    """
    ndim = len(original_shape)
    patches = np.array(patches)
    
    if ndim == 2:
        # 2D: (h, w)
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=patches.dtype)
        count = np.zeros((h, w), dtype=np.int32)
        
        patch_idx = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                if patch_idx >= len(patches):
                    break
                patch = patches[patch_idx]
                reconstructed[y:y+patch_size, x:x+patch_size] += patch
                count[y:y+patch_size, x:x+patch_size] += 1
                patch_idx += 1
        
        # Average overlapping regions
        reconstructed = reconstructed / np.maximum(count, 1)
        return reconstructed
    else:
        raise ValueError(f"Unsupported original_shape dimension: {ndim}. Expected 2D (h, w).")

def create_land_mask(bbox: List[float], reference_raster: xr.DataArray) -> xr.DataArray:
    """
    Creates a binary land mask using Natural Earth coastline data.
    
    Masks out ocean areas (0) and keeps only land areas (1).
    This significantly reduces processing time and storage by excluding
    ocean pixels that provide no training value for flood prediction.
    
    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        reference_raster: Reference xarray DataArray to match spatial properties
                         (CRS, transform, shape)
    
    Returns:
        xarray DataArray with binary land mask (1=land, 0=ocean)
        Same spatial properties as reference_raster
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
        from rasterio import features
    except ImportError:
        logger.error("geopandas required for land masking. Install with: pip install geopandas")
        raise ImportError("geopandas is required for create_land_mask")
    
    try:
        logger.info("Loading Natural Earth coastline data...")
        # Load Natural Earth land polygons (low resolution is sufficient)
        # For GeoPandas >= 1.0, datasets.get_path is deprecated, so we download directly
        try:
            # Try the deprecated method first (for older GeoPandas)
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except (AttributeError, ValueError):
            # For GeoPandas >= 1.0, download directly from Natural Earth
            import urllib.request
            import tempfile
            import os
            
            ne_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "ne_110m_admin_0_countries.zip")
            
            logger.info(f"Downloading Natural Earth data from {ne_url}...")
            urllib.request.urlretrieve(ne_url, zip_path)
            
            # Extract and load
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the shapefile
            shp_path = os.path.join(temp_dir, "ne_110m_admin_0_countries.shp")
            world = gpd.read_file(shp_path)
        
        # Create bbox polygon
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get land polygons that intersect with bbox
        land = world[world.geometry.intersects(bbox_poly)].copy()
        
        if len(land) == 0:
            logger.warning("No land polygons found in bbox. Returning all-land mask.")
            # Return mask of all ones (all land)
            if len(reference_raster.shape) >= 2:
                mask_array = np.ones(reference_raster.shape[-2:], dtype=np.uint8)
            else:
                mask_array = np.ones((reference_raster.shape[-2], reference_raster.shape[-1]), dtype=np.uint8)
            # Initialize transform for later use
            if hasattr(reference_raster, 'rio') and hasattr(reference_raster.rio, 'transform'):
                transform = reference_raster.rio.transform()
            else:
                transform = rasterio.transform.from_bounds(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    reference_raster.shape[-1], reference_raster.shape[-2]
                )
        else:
            # Reproject land polygons to match reference raster CRS if needed
            if hasattr(reference_raster, 'rio') and hasattr(reference_raster.rio, 'crs'):
                target_crs = reference_raster.rio.crs
                if target_crs is not None and hasattr(land, 'crs') and land.crs is not None:
                    # Safely compare CRS - geopandas may call .to_epsg() internally during comparison
                    # Wrap everything in try-except to catch AttributeError from .to_epsg() calls
                    try:
                        # Try to compare CRS - this may internally call .to_epsg()
                        crs_match = (land.crs == target_crs)
                        if not crs_match:
                            land = land.to_crs(target_crs)
                    except AttributeError as e:
                        # This catches 'NoneType' object has no attribute 'to_epsg'
                        # or similar errors when CRS comparison fails
                        if "'NoneType' object has no attribute 'to_epsg'" in str(e) or "to_epsg" in str(e):
                            logger.warning(f"CRS comparison failed (likely missing .to_epsg() method): {e}. Skipping reprojection.")
                        else:
                            logger.warning(f"Could not compare CRS objects: {e}. Skipping reprojection.")
                    except (ValueError, TypeError) as e:
                        # Other CRS-related errors
                        logger.warning(f"Could not reproject land polygons to target CRS: {e}. Using original CRS.")
                elif target_crs is None:
                    logger.warning("Reference raster has no CRS. Using land polygons' original CRS.")
                elif not hasattr(land, 'crs') or land.crs is None:
                    logger.warning("Land polygons have no CRS. Cannot reproject.")
            
            # Get transform and shape from reference raster
            if hasattr(reference_raster, 'rio') and hasattr(reference_raster.rio, 'transform'):
                transform = reference_raster.rio.transform()
            else:
                # Fallback: estimate transform from coords
                logger.warning("No transform found. Estimating from coordinates...")
                transform = rasterio.transform.from_bounds(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    reference_raster.shape[-1], reference_raster.shape[-2]
                )
            
            # Get shape (handle both 2D and 4D arrays)
            if len(reference_raster.shape) >= 2:
                out_shape = (reference_raster.shape[-2], reference_raster.shape[-1])
            else:
                out_shape = reference_raster.shape
            
            # Rasterize land polygons
            # 1 = land, 0 = ocean
            shapes = ((geom, 1) for geom in land.geometry)
            mask_array = features.rasterize(
                shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0,  # Fill with 0 (ocean) where no land polygons
                dtype=rasterio.uint8
            )
        
        # Create xarray DataArray with same coords as reference
        if hasattr(reference_raster, 'rio'):
            mask_xr = xr.DataArray(
                mask_array.astype(np.uint8),
                dims=['y', 'x'],
                coords={
                    'y': reference_raster.coords.get('y', np.arange(mask_array.shape[0])),
                    'x': reference_raster.coords.get('x', np.arange(mask_array.shape[1]))
                }
            )
            # Copy geospatial attributes (only if CRS exists)
            if hasattr(reference_raster.rio, 'crs') and reference_raster.rio.crs is not None:
                mask_xr.rio.write_crs(reference_raster.rio.crs, inplace=True)
            if transform is not None:
                mask_xr.rio.write_transform(transform, inplace=True)
        else:
            mask_xr = xr.DataArray(
                mask_array.astype(np.uint8),
                dims=['y', 'x']
            )
        
        # Log mask statistics
        land_pixels = np.sum(mask_array == 1)
        total_pixels = mask_array.size
        land_percent = (land_pixels / total_pixels) * 100
        logger.info(f"Land mask created: {land_pixels:,}/{total_pixels:,} pixels are land ({land_percent:.1f}%)")
        
        return mask_xr
        
    except Exception as e:
        logger.error(f"Failed to create land mask: {e}")
        logger.warning("Continuing without land mask (will process all pixels including ocean)")
        # Return all-land mask as fallback
        if len(reference_raster.shape) >= 2:
            mask_array = np.ones(reference_raster.shape[-2:], dtype=np.uint8)
        else:
            mask_array = np.ones((reference_raster.shape[-2], reference_raster.shape[-1]), dtype=np.uint8)
        
        return xr.DataArray(mask_array, dims=['y', 'x'])


def get_fractional_labels(high_res_mask, target_shape):
    """
    Creates fractional labels by upscaling a high-resolution binary mask.
    
    Args:
        high_res_mask: High-resolution binary flood mask (e.g., from Sentinel-1)
        target_shape: Target shape (height, width) for fractional labels
    
    Returns:
        xarray DataArray with fractional labels (values 0.0 to 1.0)
    """
    # This function would upscale the high-res mask to target resolution
    # and calculate the fraction of each low-res pixel that is flooded
    # For now, if already at target resolution, just return as-is
    if isinstance(high_res_mask, xr.DataArray):
        if high_res_mask.shape == target_shape:
            return high_res_mask.astype(np.float32)
        else:
            # Upscale using interpolation
            upscaled = high_res_mask.interp(
                y=np.linspace(high_res_mask.y.min(), high_res_mask.y.max(), target_shape[0]),
                x=np.linspace(high_res_mask.x.min(), high_res_mask.x.max(), target_shape[1]),
                method='nearest'
            )
            return upscaled.astype(np.float32)
    else:
        # If it's already a numpy array at target shape
        return xr.DataArray(high_res_mask.astype(np.float32), dims=['y', 'x'])
