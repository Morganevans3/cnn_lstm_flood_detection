"""Logic for STAC API (Sentinel-1/MODIS/DEM) data downloading."""
import logging
import os
from types import SimpleNamespace
import numpy as np
import pystac_client
try:
    import planetary_computer
except ImportError:
    planetary_computer = None  # Optional; required only for MODIS/DEM/PC S1
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import rioxarray
import xarray as xr
try:
    import stackstac
except ImportError:
    stackstac = None  # Optional; required only for MODIS/PC S1
import rasterio
from rasterio.enums import Resampling
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import transform_bounds, transform_geom, reproject as rasterio_reproject
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio import mask as rasterio_mask
from shapely.geometry import box as shapely_box
from shapely.geometry import mapping, shape as shapely_shape, Point

logger = logging.getLogger(__name__)

# Suppress harmless GDAL/rasterio "SHARING" warp option warning (CPLE_NotSupported)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

# Copernicus Data Space Ecosystem (CDSE) – two catalog endpoints
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"  # New STAC (Feb 2025)
CDSE_SH_CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0"  # Sentinel Hub Catalog (documented S1)
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

_cdse_token_cache: Optional[str] = None


# CDSE S3-compatible storage endpoint (data lives here, NOT on Amazon AWS)
# Full URL needed for some GDAL/boto3 configs
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
CDSE_S3_ENDPOINT_URL = "https://eodata.dataspace.copernicus.eu"


def _cdse_href_to_s3(href: str) -> str:
    """
    Convert CDSE HTTPS asset URL to s3:// path so boto3/AWSSession can access it.
    CDSE bucket is "eodata". Handles path-style (eodata/key) and virtual-host style (key).
    """
    if not isinstance(href, str) or href.startswith("s3://"):
        return href
    for prefix in (
        "https://eodata.dataspace.copernicus.eu/",
        "https://eodata.ams.dataspace.copernicus.eu/",
        "http://eodata.dataspace.copernicus.eu/",
        "https://zipper.dataspace.copernicus.eu/",
        "https://eodata.cloudferro.com/",
    ):
        if href.lower().startswith(prefix):
            path = href[len(prefix):].lstrip("/")
            if not path:
                return href
            if path.lower().startswith("eodata/"):
                path = path[7:]
            return f"s3://eodata/{path}"
    return href


def set_cdse_credentials(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    *,
    from_dict: Optional[Dict[str, str]] = None,
) -> None:
    """
    Set CDSE credentials in the environment so fetch_data_stac and download_and_load_assets
    see them (e.g. from notebook config). Call this in a notebook cell before any
    pipeline code that uses Copernicus.

    CDSE has two credential types:
    1. OAuth (client_id/client_secret or username/password): for STAC API search.
    2. S3 keys (s3_access_key/s3_secret_key): for downloading STAC assets. Get these
       from https://eodata-s3keysmanager.dataspace.copernicus.eu/ (same CDSE account).
       Data is served from eodata.dataspace.copernicus.eu - NOT Amazon AWS.

    Args:
        client_id, client_secret: OAuth for STAC search.
        username, password: Alternative OAuth (password grant).
        s3_access_key, s3_secret_key: S3-style keys for asset download (CDSE only, no AWS).
        from_dict: Optional dict with keys CDSE_CLIENT_ID, CDSE_CLIENT_SECRET,
            CDSE_USERNAME, CDSE_PASSWORD, CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY.
    """
    if from_dict:
        for k, v in from_dict.items():
            if v is None:
                continue
            key = k.upper() if not k.startswith("CDSE_") else k
            if not key.startswith("CDSE_"):
                key = f"CDSE_{key}"
            os.environ[key] = str(v)
    if client_id is not None:
        os.environ["CDSE_CLIENT_ID"] = str(client_id)
    if client_secret is not None:
        os.environ["CDSE_CLIENT_SECRET"] = str(client_secret)
    if username is not None:
        os.environ["CDSE_USERNAME"] = str(username)
    if password is not None:
        os.environ["CDSE_PASSWORD"] = str(password)
    if s3_access_key is not None:
        os.environ["CDSE_S3_ACCESS_KEY"] = str(s3_access_key)
    if s3_secret_key is not None:
        os.environ["CDSE_S3_SECRET_KEY"] = str(s3_secret_key)


def _get_cdse_s3_credentials(credentials: Optional[Dict[str, str]] = None) -> Optional[tuple]:
    """
    Get (access_key, secret_key) for CDSE S3, from credentials dict or os.environ.
    Returns None if not available.
    """
    def get(key: str) -> Optional[str]:
        if credentials:
            for k in (key, key.replace("CDSE_", ""), f"CDSE_{key}"):
                if k in credentials and credentials[k]:
                    return str(credentials[k]).strip()
        return (os.getenv(key) or os.getenv(key.replace("CDSE_", "")) or "").strip() or None

    access = get("CDSE_S3_ACCESS_KEY")
    secret = get("CDSE_S3_SECRET_KEY")
    if not access or not secret:
        return None
    return (access, secret)


def _configure_cdse_s3_env(credentials: Optional[Dict[str, str]] = None) -> bool:
    """
    Configure GDAL/rasterio to use CDSE S3-compatible storage (eodata.dataspace.copernicus.eu)
    instead of Amazon AWS. Must be called before opening CDSE STAC asset URLs.

    Sets both hostname and full URL for AWS_S3_ENDPOINT (GDAL varies by version).
    Returns True if CDSE S3 credentials were set, False otherwise.
    """
    creds = _get_cdse_s3_credentials(credentials)
    if not creds:
        return False
    access, secret = creds
    os.environ["AWS_S3_ENDPOINT"] = CDSE_S3_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = access
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret
    os.environ["AWS_HTTPS"] = "YES"
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE", "YES")
    return True


def _cdse_rasterio_env(
    credentials: Optional[Dict[str, str]] = None,
    cdse_token: Optional[str] = None,
):
    """
    Return rasterio.Env configured for CDSE (eodata.dataspace.copernicus.eu).
    Uses boto3 via AWSSession for S3-style access. When both S3 creds and token
    are present, also sets GDAL_HTTP_HEADER for HTTPS hrefs that need Bearer auth.
    """
    creds = _get_cdse_s3_credentials(credentials)
    env_kw: Dict[str, Any] = {}
    if creds:
        access, secret = creds
        from rasterio.session import AWSSession
        session = AWSSession(
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            endpoint_url=CDSE_S3_ENDPOINT_URL,
            region_name="default",
        )
        env_kw["session"] = session
    if cdse_token:
        env_kw["GDAL_HTTP_HEADER"] = f"Authorization: Bearer {cdse_token}"
    return rasterio.Env(**env_kw) if env_kw else rasterio.Env()


def _get_cdse_token(credentials: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Get OAuth2 access token for Copernicus Data Space Ecosystem.
    Uses CDSE_USERNAME + CDSE_PASSWORD (password grant) or
    CDSE_CLIENT_ID + CDSE_CLIENT_SECRET (client credentials).
    Caches token for the session when using env vars only.

    Args:
        credentials: Optional dict with keys CDSE_USERNAME, CDSE_PASSWORD and/or
            CDSE_CLIENT_ID, CDSE_CLIENT_SECRET. When provided, these override
            environment variables (e.g. from notebook config). Not cached.
    """
    global _cdse_token_cache
    use_env = credentials is None
    if use_env and _cdse_token_cache is not None:
        return _cdse_token_cache

    def get(key: str, default: str = "") -> str:
        if credentials is not None:
            alt = key.replace("CDSE_", "") if key.startswith("CDSE_") else f"CDSE_{key}"
            for k in (key, alt):
                if k in credentials and credentials[k] not in (None, ""):
                    return str(credentials[k]).strip()
        return (os.getenv(key) or default).strip()

    username = get("CDSE_USERNAME")
    password = get("CDSE_PASSWORD")
    client_id = get("CDSE_CLIENT_ID") or "cdse-public"
    client_secret = get("CDSE_CLIENT_SECRET")

    if username and password:
        data = {
            "grant_type": "password",
            "client_id": client_id,
            "username": username,
            "password": password,
        }
        if client_secret:
            data["client_secret"] = client_secret
    elif client_id and client_secret:
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    else:
        if use_env:
            logger.warning(
                "CDSE credentials not set. Set CDSE_USERNAME and CDSE_PASSWORD, or "
                "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET, to use Copernicus Sentinel-1."
            )
        return None
    try:
        r = requests.post(CDSE_TOKEN_URL, data=data, timeout=30)
        r.raise_for_status()
        token = r.json().get("access_token")
        if use_env:
            _cdse_token_cache = token
        return token
    except Exception as e:
        logger.warning(f"Failed to get CDSE token: {e}")
        return None


def get_catalog_copernicus(
    use_signing: bool = True,
    credentials: Optional[Dict[str, str]] = None,
) -> Optional["pystac_client.Client"]:
    """
    Connect to Copernicus Data Space Ecosystem STAC API.
    Requires CDSE_USERNAME/CDSE_PASSWORD or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET
    (env vars or pass via credentials dict).
    When use_signing=True, adds Bearer token to requests (needed for asset access).
    """
    token = _get_cdse_token(credentials)
    if not token:
        return None

    def add_bearer(request):
        request.headers["Authorization"] = f"Bearer {token}"

    modifier = add_bearer if use_signing else None
    try:
        return pystac_client.Client.open(CDSE_STAC_URL, modifier=modifier)
    except Exception as e:
        logger.warning(f"Failed to open CDSE STAC catalog: {e}")
        return None


def _read_s1_vv_window_with_proj_metadata(
    s1_item_signed: Any,
    bbox: List[float],
    label: str = "Sentinel-1",
    cdse_token: Optional[str] = None,
    cdse_credentials: Optional[Dict[str, str]] = None,
    target_resolution_m: Optional[float] = None,
) -> xr.DataArray:
    """
    Read a bbox window from a Sentinel-1 COG even when the raster has no CRS tag.

    stackstac/rasterio can fail with:
      AttributeError: 'NoneType' object has no attribute 'to_epsg'
    when ds.crs is None. Many Sentinel-1 STAC items still provide enough info via
    proj:* metadata to compute a window and attach CRS/transform ourselves.
    """
    if "vv" not in getattr(s1_item_signed, "assets", {}):
        raise ValueError(f"{label} item missing 'vv' asset")

    vv_href = s1_item_signed.assets["vv"].href
    proj_epsg = None
    proj_transform = None

    try:
        props = getattr(s1_item_signed, "properties", {}) or {}
        proj_epsg = props.get("proj:epsg")
        proj_transform = props.get("proj:transform")
    except Exception:
        proj_epsg = None
        proj_transform = None

    if not proj_epsg or not proj_transform or len(proj_transform) < 6:
        raise ValueError(f"{label} missing proj:epsg/proj:transform; cannot window-read safely")

    src_crs = CRS.from_epsg(int(proj_epsg))
    base_transform = Affine(*proj_transform[:6])

    # Convert WGS84 bbox to source CRS bbox, then compute a pixel window
    minx, miny, maxx, maxy = bbox
    if src_crs != CRS.from_epsg(4326):
        minx, miny, maxx, maxy = transform_bounds(
            CRS.from_epsg(4326), src_crs, minx, miny, maxx, maxy, densify_pts=21
        )

    win = from_bounds(minx, miny, maxx, maxy, transform=base_transform)
    win = win.round_offsets().round_lengths()

    if win.width <= 0 or win.height <= 0:
        raise ValueError(f"No data found in bounds for {label} window")

    # Optional: decimate during read to avoid loading full native-res array (~12GB per 5° tile)
    out_shape = None
    if target_resolution_m is not None and target_resolution_m > 0:
        native_res = abs(base_transform.a)
        if native_res > 0:
            scale = target_resolution_m / native_res
            h_out = max(1, int(round(win.height / scale)))
            w_out = max(1, int(round(win.width / scale)))
            out_shape = (h_out, w_out)

    open_href = _cdse_href_to_s3(vv_href)
    with _cdse_rasterio_env(cdse_credentials, cdse_token):
        with rasterio.open(open_href) as src:
            if out_shape:
                data = src.read(1, window=win, out_shape=out_shape, resampling=Resampling.average)
            else:
                data = src.read(1, window=win)

    if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(f"No data found in bounds for {label} window")

    w_transform = window_transform(win, base_transform)
    height, width = data.shape
    if out_shape:
        # Adjust transform for decimated output (pixel size increased by scale factor)
        scale_y = win.height / height
        scale_x = win.width / width
        w_transform = Affine(
            w_transform.a * scale_x, w_transform.b, w_transform.c,
            w_transform.d, w_transform.e * scale_y, w_transform.f,
        )

    # Build pixel-center coordinates
    xs = w_transform.c + (0.5 + np.arange(width)) * w_transform.a + (0.5) * w_transform.b
    ys = w_transform.f + (0.5 + np.arange(height)) * w_transform.e + (0.5) * w_transform.d

    da = xr.DataArray(data, dims=("y", "x"), coords={"y": ys, "x": xs})
    da = da.rio.write_transform(w_transform, inplace=False)
    da = da.rio.write_crs(src_crs, inplace=False)
    return da


def _read_s1_vv_window_from_file(
    vv_href: str,
    bbox: List[float],
    label: str = "Sentinel-1",
    cdse_token: Optional[str] = None,
    cdse_credentials: Optional[Dict[str, str]] = None,
    target_resolution_m: Optional[float] = None,
) -> Optional[xr.DataArray]:
    """
    Read a bbox window from a Sentinel-1 COG using CRS/transform read from the file.

    Use when the STAC item has no proj:epsg/proj:transform; many COGs still have
    CRS and transform in the GeoTIFF tags.
    """
    open_href = _cdse_href_to_s3(vv_href)
    with _cdse_rasterio_env(cdse_credentials, cdse_token):
        with rasterio.open(open_href) as src:
            src_crs = src.crs
            base_transform = src.transform
            if src_crs is None or base_transform is None:
                return None

            minx, miny, maxx, maxy = bbox
            if src_crs != CRS.from_epsg(4326):
                minx, miny, maxx, maxy = transform_bounds(
                    CRS.from_epsg(4326), src_crs, minx, miny, maxx, maxy, densify_pts=21
                )
            win = from_bounds(minx, miny, maxx, maxy, transform=base_transform)
            win = win.round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0:
                return None
            out_shape = None
            if target_resolution_m is not None and target_resolution_m > 0:
                native_res = abs(base_transform.a)
                if native_res > 0:
                    scale = target_resolution_m / native_res
                    h_out = max(1, int(round(win.height / scale)))
                    w_out = max(1, int(round(win.width / scale)))
                    out_shape = (h_out, w_out)
            if out_shape:
                data = src.read(1, window=win, out_shape=out_shape, resampling=Resampling.average)
            else:
                data = src.read(1, window=win)

    if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
        return None

    w_transform = window_transform(win, base_transform)
    height, width = data.shape
    if out_shape:
        scale_y = win.height / height
        scale_x = win.width / width
        w_transform = Affine(
            w_transform.a * scale_x, w_transform.b, w_transform.c,
            w_transform.d, w_transform.e * scale_y, w_transform.f,
        )
    xs = w_transform.c + (0.5 + np.arange(width)) * w_transform.a + (0.5) * w_transform.b
    ys = w_transform.f + (0.5 + np.arange(height)) * w_transform.e + (0.5) * w_transform.d
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": ys, "x": xs})
    da = da.rio.write_transform(w_transform, inplace=False)
    da = da.rio.write_crs(src_crs, inplace=False)
    return da


def _read_s1_vv_full_then_clip(
    vv_href: str,
    bbox: List[float],
    label: str = "Sentinel-1",
    item_geometry: Any = None,
    cdse_token: Optional[str] = None,
    cdse_credentials: Optional[Dict[str, str]] = None,
    target_resolution_m: Optional[float] = None,
) -> xr.DataArray:
    """
    Load Sentinel-1 VV asset and clip to bbox using rasterio.mask. Use when windowed
    read is not possible (no CRS/transform on STAC item or in file). Optionally uses
    the STAC item geometry (intersected with bbox) so the clip shape matches the
    scene footprint and avoids "No data found in bounds" when COG extent differs.
    """
    minx, miny, maxx, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_geom = shapely_box(minx, miny, maxx, maxy)

    # Prefer clip shape = (item footprint ∩ bbox) so we clip to scene overlap only
    if item_geometry is not None:
        try:
            item_poly = shapely_shape(item_geometry)
            clip_poly = item_poly.intersection(bbox_geom)
            if clip_poly.is_empty or (hasattr(clip_poly, "area") and clip_poly.area < 1e-10):
                clip_poly = bbox_geom
            elif clip_poly.geom_type == "MultiPolygon":
                clip_poly = max(clip_poly.geoms, key=lambda g: g.area)
            if clip_poly.geom_type != "Polygon":
                clip_poly = bbox_geom
            geom_wgs84 = mapping(clip_poly)
        except Exception:
            geom_wgs84 = mapping(bbox_geom)
    else:
        geom_wgs84 = mapping(bbox_geom)

    open_href = _cdse_href_to_s3(vv_href)
    with _cdse_rasterio_env(cdse_credentials, cdse_token):
        with rasterio.open(open_href) as src:
            if src.crs is None:
                # COG has no CRS: load full raster, reproject to WGS84, then clip in 4326
                return _read_s1_full_reproject_then_clip(
                    vv_href, bbox, label, item_geometry=item_geometry,
                    cdse_token=cdse_token, cdse_credentials=cdse_credentials,
                    target_resolution_m=target_resolution_m,
                )
            try:
                geom_raster_crs = transform_geom(
                    CRS.from_epsg(4326), src.crs, geom_wgs84, densify_pts=21
                )
            except Exception as e:
                raise ValueError(f"Could not transform bbox to raster CRS: {e}")
            try:
                data, out_transform = rasterio_mask.mask(
                    src, [geom_raster_crs], crop=True, filled=False, indexes=[1]
                )
            except ValueError as e:
                if "No data found in bounds" in str(e) or "does not overlap" in str(e).lower():
                    # Fallback: load full raster, reproject to WGS84, clip there
                    logger.info("rasterio.mask reported no overlap; trying full read + reproject + clip.")
                    return _read_s1_full_reproject_then_clip(
                        vv_href, bbox, label, item_geometry=item_geometry,
                        cdse_token=cdse_token, cdse_credentials=cdse_credentials,
                        target_resolution_m=target_resolution_m,
                    )
                raise
            if data.size == 0 or data.shape[-2] == 0 or data.shape[-1] == 0:
                raise ValueError(f"Could not clip {label} to bbox: clip result is empty.")
            src_crs = src.crs
            height, width = data.shape[-2], data.shape[-1]

    # Build pixel-center coordinates and xarray DataArray (match other S1 load paths)
    w_transform = out_transform
    xs = w_transform.c + (0.5 + np.arange(width)) * w_transform.a + (0.5) * w_transform.b
    ys = w_transform.f + (0.5 + np.arange(height)) * w_transform.e + (0.5) * w_transform.d
    data_2d = np.squeeze(data)
    if data_2d.ndim != 2:
        data_2d = data_2d[0]
    da = xr.DataArray(data_2d, dims=("y", "x"), coords={"y": ys, "x": xs})
    da = da.rio.write_transform(w_transform, inplace=False)
    da = da.rio.write_crs(src_crs, inplace=False)
    return da


def _utm_epsg_from_bbox(bbox: List[float]) -> int:
    """Guess UTM EPSG (WGS 84 / UTM zone N or S) from a WGS84 bbox [min_lon, min_lat, max_lon, max_lat]."""
    lon_c = (bbox[0] + bbox[2]) / 2.0
    lat_c = (bbox[1] + bbox[3]) / 2.0
    zone = int((lon_c + 180) / 6) + 1
    zone = max(1, min(60, zone))
    if lat_c < 0:
        return 32700 + zone  # UTM South (e.g. 32748 for zone 48S)
    return 32600 + zone  # UTM North


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Guess UTM EPSG from a single (lon, lat) point (e.g. scene centroid)."""
    zone = int((lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    if lat < 0:
        return 32700 + zone
    return 32600 + zone


def _centroid_from_geometry(geom: Any) -> Optional[tuple]:
    """Get (lon, lat) centroid from GeoJSON-like geometry, or None."""
    try:
        poly = shapely_shape(geom)
        c = poly.centroid
        return (float(c.x), float(c.y))
    except Exception:
        return None


def _bounds_from_geometry(geom: Any) -> Optional[tuple]:
    """Get (minx, miny, maxx, maxy) in WGS84 from GeoJSON-like geometry, or None."""
    try:
        poly = shapely_shape(geom)
        return poly.bounds
    except Exception:
        return None


def _is_identity_transform(t: Any) -> bool:
    """True if transform is identity (pixel coords = map coords)."""
    if t is None:
        return True
    return (
        abs(getattr(t, "a", 1) - 1) < 0.01
        and abs(getattr(t, "e", 1) - 1) < 0.01
        and abs(getattr(t, "b", 0)) < 0.01
        and abs(getattr(t, "d", 0)) < 0.01
        and abs(getattr(t, "c", 0)) < 0.01
        and abs(getattr(t, "f", 0)) < 0.01
    )


def _utm_zones_to_try(bbox: List[float], primary_epsg: int) -> List[int]:
    """Return list of EPSG codes (UTM zones) to try when primary gives all-NaN (S1 scene may span different zone).
    Indonesia spans zones 46-52; try primary and neighbors."""
    lat_c = (bbox[1] + bbox[3]) / 2.0
    base = 32700 if lat_c < 0 else 32600
    primary_zone = primary_epsg - base
    zone_nums = [
        primary_zone,
        primary_zone - 1,
        primary_zone + 1,
        primary_zone - 2,
        primary_zone + 2,
        primary_zone - 3,
        primary_zone + 3,
    ]
    return [base + max(1, min(60, z)) for z in zone_nums]


def _infer_utm_zone_from_raster(
    src_transform: Any,
    width: int,
    height: int,
    item_geometry: Any,
) -> Optional[int]:
    """
    Infer UTM zone by testing: raster center (x,y) in each zone → (lon,lat).
    Return the zone whose (lon,lat) falls inside item_geometry. Indonesia: zones 46-52.
    """
    if src_transform is None or width <= 0 or height <= 0:
        return None
    try:
        from pyproj import Transformer
    except ImportError:
        return None
    # Raster center in pixel coords (col, row)
    cx, cy = width / 2.0, height / 2.0
    # To map coords: affine * (col, row)
    mx = src_transform.c + cx * src_transform.a + cy * src_transform.b
    my = src_transform.f + cx * src_transform.d + cy * src_transform.e
    # Test each UTM zone (Indonesia south: 32746-32752)
    try:
        poly = shapely_shape(item_geometry) if item_geometry else None
    except Exception:
        poly = None
    for zone in range(46, 53):
        epsg = 32700 + zone
        try:
            trans = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            lon, lat = trans.transform(mx, my)
            if poly is not None and poly.contains(Point(lon, lat)):
                return epsg
            if 95 <= lon <= 141 and -15 <= lat <= 10:
                return epsg
        except Exception:
            continue
    return None


def _read_s1_full_reproject_then_clip(
    vv_href: str,
    bbox: List[float],
    label: str = "Sentinel-1",
    item_geometry: Any = None,
    cdse_token: Optional[str] = None,
    cdse_credentials: Optional[Dict[str, str]] = None,
    target_resolution_m: Optional[float] = None,
) -> xr.DataArray:
    """
    Reproject source raster directly into a destination that is exactly our bbox
    in WGS84. Avoids clip_box entirely so we never hit "No data found in bounds".
    If the COG has no CRS (src.crs is None), assumes UTM from scene centroid (preferred)
    or bbox center. When the primary zone yields all-NaN, tries adjacent UTM zones.
    Can be slow and memory-heavy for large scenes (reads full band).
    Use target_resolution_m=500 to reduce output size ~50x (500m vs 11m).
    """
    from rasterio.enums import Resampling
    minx, miny, maxx, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    # 500m = 0.00449 deg; 11m = 0.0001 deg. 500m reduces memory ~50x for 5° tiles.
    res_deg = 0.00449 if (target_resolution_m and target_resolution_m >= 400) else 0.0001
    dst_width = max(1, int(round((maxx - minx) / res_deg)))
    dst_height = max(1, int(round((maxy - miny) / res_deg)))
    dst_transform = transform_from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)

    open_href = _cdse_href_to_s3(vv_href)
    with _cdse_rasterio_env(cdse_credentials, cdse_token):
        with rasterio.open(open_href) as src:
            src_data = src.read(1)
            src_crs = src.crs
            src_transform = src.transform
            src_nodata = getattr(src, "nodata", None)
            src_width = src.width
            src_height = src.height

    # Log if source has valid data (helps debug all-NaN)
    n_finite = int(np.sum(np.isfinite(src_data)))
    if n_finite == 0:
        logger.warning(
            f"{label} source raster has no finite values before reproject. "
            f"dtype={src_data.dtype}, shape={src_data.shape}, nodata={src_nodata}, "
            f"transform_scale={abs(src_transform.a) if src_transform else None}."
        )

    effective_transform = src_transform
    zones_to_try: List[Any] = []

    if src_crs is not None:
        zones_to_try = [src_crs]
    elif _is_identity_transform(src_transform) and item_geometry:
        bounds_from_geom = _bounds_from_geometry(item_geometry)
        if bounds_from_geom:
            minx_g, miny_g, maxx_g, maxy_g = bounds_from_geom
            effective_transform = transform_from_bounds(
                minx_g, miny_g, maxx_g, maxy_g, src_width, src_height
            )
            zones_to_try = [CRS.from_epsg(4326)]
            logger.info(
                f"{label} COG has identity transform; using scene geometry bounds (WGS84) "
                f"[{minx_g:.2f},{miny_g:.2f},{maxx_g:.2f},{maxy_g:.2f}]."
            )

    if not zones_to_try:
        # Fallback: infer UTM zone from raster center or centroid
        inferred_epsg = _infer_utm_zone_from_raster(
            src_transform, src_width, src_height, item_geometry
        )
        if inferred_epsg is not None:
            guessed_epsg = inferred_epsg
            logger.info(
                f"{label} COG has no CRS; inferred EPSG:{guessed_epsg} from raster center + scene geometry."
            )
        else:
            centroid = _centroid_from_geometry(item_geometry)
            if centroid:
                guessed_epsg = _utm_epsg_from_lonlat(centroid[0], centroid[1])
                logger.info(
                    f"{label} COG has no CRS; using scene centroid -> EPSG:{guessed_epsg}."
                )
            else:
                guessed_epsg = _utm_epsg_from_bbox(bbox)
                logger.info(
                    f"{label} COG has no CRS; using bbox center -> EPSG:{guessed_epsg}."
                )
        epsg_list = list(dict.fromkeys(_utm_zones_to_try(bbox, guessed_epsg)))
        zones_to_try = [CRS.from_epsg(z) for z in epsg_list]
        scale = abs(src_transform.a) if src_transform else 1.0
        if scale < 0.01:
            zones_to_try.insert(0, CRS.from_epsg(4326))

    def _reproject_with_crs(try_crs: CRS, src_tf: Any = None) -> np.ndarray:
        tf = src_tf if src_tf is not None else src_transform
        dst = np.empty((dst_height, dst_width), dtype=np.float32)
        dst[:] = np.nan
        rasterio_reproject(
            source=src_data,
            destination=dst,
            src_transform=tf,
            src_crs=try_crs,
            dst_transform=dst_transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
        return dst

    dst_array = _reproject_with_crs(zones_to_try[0], effective_transform)
    n_valid = int(np.sum(np.isfinite(dst_array)))
    min_valid = max(100, int(0.01 * dst_array.size))

    if n_valid < min_valid and len(zones_to_try) > 1:
        for try_crs in zones_to_try[1:]:
            candidate = _reproject_with_crs(try_crs, effective_transform)
            n = int(np.sum(np.isfinite(candidate)))
            if n >= min_valid:
                dst_array = candidate
                logger.info(
                    f"{label} all-NaN with primary CRS; EPSG:{try_crs.to_epsg()} gave {n} valid pixels."
                )
                break

    # Build xarray DataArray (match other S1 load paths)
    xs = dst_transform.c + (0.5 + np.arange(dst_width)) * dst_transform.a
    ys = dst_transform.f + (0.5 + np.arange(dst_height)) * dst_transform.e
    da = xr.DataArray(dst_array, dims=("y", "x"), coords={"y": ys, "x": xs})
    da = da.rio.write_transform(dst_transform, inplace=False)
    da = da.rio.write_crs(CRS.from_epsg(4326), inplace=False)
    return da


def _clip_box_wgs84(ds: xr.DataArray, bbox: List[float], label: str, strict: bool = False) -> xr.DataArray:
    """
    Clip a raster to a WGS84 bbox, transforming bbox to dataset CRS if needed.

    Many Planetary Computer rasters (e.g., Sentinel-1) are in projected UTM CRS.
    Passing a lon/lat bbox directly to clip_box will raise "No data found in bounds".
    """
    try:
        if not hasattr(ds, "rio") or ds.rio.crs is None:
            # Best effort: attempt clip assuming bbox is already in ds CRS
            return ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        src_crs = CRS.from_epsg(4326)
        dst_crs = ds.rio.crs

        minx, miny, maxx, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
        if dst_crs != src_crs:
            minx, miny, maxx, maxy = transform_bounds(
                src_crs, dst_crs, minx, miny, maxx, maxy, densify_pts=21
            )

        return ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    except Exception as e:
        # For some assets (notably Sentinel-1 labels), falling back to full extent can be
        # catastrophically slow. In those cases we prefer to fail fast and skip the date.
        if strict:
            raise ValueError(f"Could not clip {label} to bbox: {e}")
        logger.warning(f"Could not clip {label} to bbox: {e}. Using full extent.")
        return ds

def get_catalog(use_signing=True):
    """
    Connects to the Planetary Computer STAC API with optional SAS token signing.
    
    Planetary Computer data is anonymously accessible - no subscription key required.
    The planetary_computer package automatically signs URLs with SAS tokens when needed.
    
    Optional: Set PC_SDK_SUBSCRIPTION_KEY environment variable for:
    - Better rate limits
    - Longer token expiry times
    - Priority access during high traffic
    
    Args:
        use_signing: If True, use sign_inplace modifier. If False, get items without signing
                    (signing will happen later when accessing asset URLs)
    
    Returns:
        pystac_client.Client: STAC client for Planetary Computer
    
    Note:
        Uses planetary_computer.sign_inplace to automatically sign asset URLs
        with SAS tokens when accessing Azure Blob Storage data.
        The signing happens automatically when you access item.assets['band'].href
    """
    # Optionally set subscription key if provided (improves rate limits)
    subscription_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY")
    if subscription_key:
        planetary_computer.settings.set_subscription_key(subscription_key)
        logger.info("Using Planetary Computer subscription key for enhanced rate limits")
    else:
        logger.info("Accessing Planetary Computer anonymously (subscription key optional)")
    
    # The sign_inplace modifier automatically signs URLs with SAS tokens
    if use_signing:
        return pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    else:
        # Open without modifier - we'll sign URLs manually when needed
        return pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )


def fetch_sentinel1(
    bbox: List[float],
    flood_date: str,
    days_window: int = 8,
    collections: List[str] = None,
):
    """
    Searches for Sentinel-1 SAR data around the flood date.

    Tries each collection in order (e.g. RTC then GRD). RTC is terrain-corrected
    and often has CRS in the COG; GRD may have missing CRS and require fallbacks.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        flood_date: Flood event date in 'YYYY-MM-DD' format
        days_window: Days to search before/after flood date (default: 8)
        collections: STAC collection IDs to try in order (default: ["sentinel-1-grd"])

    Returns:
        List: List of Sentinel-1 STAC items found in the search window
    """
    if collections is None:
        collections = ["sentinel-1-grd"]
    catalog = get_catalog()
    flood_dt = datetime.strptime(flood_date, "%Y-%m-%d")
    start_dt = flood_dt - timedelta(days=days_window)
    end_dt = flood_dt + timedelta(days=days_window)
    datetime_range = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"

    for collection_id in collections:
        try:
            kwargs = dict(
                collections=[collection_id],
                bbox=bbox,
                datetime=datetime_range,
            )
            if "grd" in collection_id.lower():
                kwargs["query"] = {"sar:instrument_mode": {"eq": "IW"}}
            search = catalog.search(**kwargs)
            if search is None:
                continue
            items = []
            for item in search.items():
                props = getattr(item, "properties", {}) or {}
                has_proj = bool(props.get("proj:epsg")) and bool(props.get("proj:transform"))
                if has_proj:
                    items.append(item)
                    break
            if not items:
                for item in search.items():
                    items.append(item)
                    break
            if items:
                logger.info(f"Using Sentinel-1 from collection: {collection_id}")
                return items
        except Exception as e:
            logger.debug(f"Sentinel-1 collection {collection_id}: {e}")
            continue
    return []


def _cdse_search_post(
    token: str,
    bbox: List[float],
    datetime_range: str,
    collection_id: str,
    prefer_cog_safe: bool = True,
) -> List[Any]:
    """
    Direct POST to CDSE STAC /search; returns list of item-like objects (id, assets, geometry).
    When prefer_cog_safe=True, sorts results so COG_SAFE items (id contains '_COG') come first.
    COG_SAFE = Cloud Optimized GeoTIFF format, better for streaming/partial reads than original GRD.
    """
    def _has_vv(assets: dict) -> bool:
        return any(k.lower() == "vv" for k in assets)

    payload = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": 10,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    for base_url in [CDSE_STAC_URL, CDSE_SH_CATALOG_URL]:
        url = f"{base_url.rstrip('/')}/search"
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            features = data.get("features") or []
            items = []
            for f in features:
                assets = f.get("assets") or {}
                item_assets = {}
                for k, v in assets.items():
                    href = v.get("href")
                    if not href:
                        continue
                    # Prefer S3 alternate when available (CDSE uses alternate-assets extension)
                    alt = v.get("alternate") or {}
                    s3_alt = alt.get("s3") or alt.get("S3") or {}
                    if s3_alt.get("href"):
                        href = s3_alt["href"]
                    item_assets[k] = SimpleNamespace(href=href)
                if not _has_vv(item_assets):
                    continue
                items.append(
                    SimpleNamespace(
                        id=f.get("id", ""),
                        geometry=f.get("geometry"),
                        assets=item_assets,
                        properties=f.get("properties") or {},
                    )
                )
            if prefer_cog_safe and items:
                items.sort(key=lambda x: (0 if "_COG" in getattr(x, "id", "") else 1))
            if items:
                return items
        except Exception as e:
            logger.debug("CDSE search %s for %s: %s", url, collection_id, e)
            continue
    return []


def fetch_sentinel1_copernicus(
    bbox: List[float],
    flood_date: str,
    days_window: int = 14,
    cdse_credentials: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """
    Search for Sentinel-1 GRD data from Copernicus Data Space Ecosystem (CDSE) STAC.
    Prefers COG_SAFE products (Cloud Optimized GeoTIFF) when available - same metadata as GRD,
    but -cog.tiff measurements enable efficient streaming. CDSE converts the archive gradually.
    Requires CDSE_USERNAME + CDSE_PASSWORD or CDSE_CLIENT_ID + CDSE_CLIENT_SECRET
    (env vars or pass via cdse_credentials, e.g. from notebook config).
    """
    token = _get_cdse_token(cdse_credentials)
    if not token:
        logger.warning("Copernicus catalog not available (missing credentials).")
        return []
    flood_dt = datetime.strptime(flood_date, "%Y-%m-%d")
    start_dt = flood_dt - timedelta(days=days_window)
    end_dt = flood_dt + timedelta(days=days_window)
    datetime_range = (
        f"{start_dt.strftime('%Y-%m-%d')}T00:00:00Z/{end_dt.strftime('%Y-%m-%d')}T23:59:59Z"
    )
    # Try multiple collection ids (CDSE STAC naming can vary)
    for collection_id in ["sentinel-1-grd", "SENTINEL-1", "sentinel-1-rtc"]:
        items = _cdse_search_post(token, bbox, datetime_range, collection_id, prefer_cog_safe=True)
        if items:
            chosen = items[0]
            cog_note = " (COG_SAFE)" if "_COG" in getattr(chosen, "id", "") else ""
            logger.info(
                "Using Sentinel-1 from Copernicus Data Space Ecosystem (collection %s)%s.",
                collection_id,
                cog_note,
            )
            return [chosen]
    # Fallback: pystac_client catalog search (modifier may behave differently)
    catalog = get_catalog_copernicus(use_signing=True, credentials=cdse_credentials)
    if catalog:
        for collection_id in ["sentinel-1-grd", "SENTINEL-1", "sentinel-1-rtc"]:
            try:
                search = catalog.search(
                    collections=[collection_id],
                    bbox=bbox,
                    datetime=datetime_range,
                )
                if search is None:
                    continue
                for item in search.items():
                    logger.info(
                        "Using Sentinel-1 from Copernicus (pystac_client, collection %s).",
                        collection_id,
                    )
                    return [item]
            except Exception as e:
                logger.debug("pystac_client search for %s failed: %s", collection_id, e)
    logger.info(
        "Copernicus returned no Sentinel-1 for bbox=%s datetime=%s. "
        "Check https://browser.stac.dataspace.copernicus.eu and CDSE credentials.",
        bbox,
        datetime_range,
    )
    return []


def fetch_modis(bbox: List[float], flood_date: str, days_back: int = 80):
    """
    Searches for MODIS 8-day composite data leading up to the flood date.
    
    Retrieves MODIS MOD09A1 data for the time series required by the model.
    Typically needs ~10 consecutive 8-day composites, so searches back 80 days
    to ensure sufficient data coverage.
    
    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        flood_date: Flood event date in 'YYYY-MM-DD' format
        days_back: Number of days to look back from flood date (default: 80)
    
    Returns:
        List: List of MODIS STAC items found in the search window
    """
    # Use catalog with sign_inplace so asset hrefs are signed when accessed
    catalog = get_catalog(use_signing=True)
    end_dt = datetime.strptime(flood_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=days_back)
    
    # Collection ID confirmed: modis-09A1-061
    # Reference: https://planetarycomputer.microsoft.com/api/stac/v1/collections/modis-09A1-061
    search = catalog.search(
        collections=["modis-09A1-061"],  # MODIS Terra Surface Reflectance 8-Day
        bbox=bbox,
        datetime=f"{start_dt.strftime('%Y-%m-%d')}/{flood_date}"
    )
    
    # Get items without signing - we'll sign URLs later when accessing assets
    items = []
    try:
        for item in search.items():
            items.append(item)
            # Limit to first 20 items (we only need ~10 for sequence)
            if len(items) >= 20:
                break
    except Exception as e:
        logger.error(f"Error accessing MODIS items: {e}")
        return []
    
    if len(items) == 0:
        logger.warning("No MODIS items found.")
    
    return items


def fetch_dem(bbox: List[float]):
    """
    Searches for DEM elevation data.
    
    Tries multiple DEM collections to find available data.
    This is a static dataset, so no date parameter is needed.
    
    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
    
    Returns:
        List: List of DEM STAC items (typically 1 item per bbox)
    
    Note:
        Tries COP-DEM GLO-30 first, falls back to other DEM sources if needed
    """
    catalog = get_catalog()
    
    # Try different DEM collection names
    dem_collections = [
        "cop-dem-glo-30",  # Copernicus DEM Global 30m
        "nasadem",  # NASA DEM
        "srtm",  # SRTM
    ]
    
    for collection in dem_collections:
        try:
            search = catalog.search(
                collections=[collection],
                bbox=bbox,
                limit=1  # Just test if it works
            )
            # Try to get first item without signing all items
            items = []
            for item in search.items():
                items.append(item)
                break  # Just need one to test
            
            if items:
                logger.info(f"Successfully found DEM data using collection: {collection}")
                # Get all items for this collection
                full_search = catalog.search(collections=[collection], bbox=bbox)
                all_items = []
                for item in full_search.items():
                    all_items.append(item)
                    if len(all_items) >= 5:  # Limit to avoid issues
                        break
                return all_items
        except Exception as e:
            logger.debug(f"Collection {collection} failed: {e}")
            continue
    
    # If all fail, return empty list with warning
    logger.warning(f"No DEM data found in any collection. Searched: {dem_collections}")
    logger.warning("You may need to check available DEM collections at:")
    logger.warning("https://planetarycomputer.microsoft.com/datasets")
    return []


def fetch_data_stac(
    bbox: List[float],
    flood_date: str,
    s1_days_window: int = 14,
    modis_days_back: int = 80,
    s1_collections: List[str] = None,
    use_copernicus_s1: bool = True,
    cdse_credentials: Optional[Dict[str, str]] = None,
    s1_only: bool = False,
) -> Dict[str, Any]:
    """
    Searches for satellite data: Sentinel-1 (and optionally MODIS, DEM).

    Sentinel-1: by default uses only Copernicus Data Space Ecosystem (CDSE); requires
    CDSE credentials (env vars or pass cdse_credentials from notebook/config).
    Set use_copernicus_s1=False to use Microsoft Planetary Computer instead.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        flood_date: Flood event date in 'YYYY-MM-DD' format
        s1_days_window: Days to search before/after flood date for Sentinel-1 (default: 14)
        modis_days_back: Days to look back from flood date for MODIS (default: 80)
        s1_collections: Sentinel-1 STAC collection IDs (Planetary Computer only; default: ["sentinel-1-grd"])
        use_copernicus_s1: If True, get Sentinel-1 from Copernicus Data Space Ecosystem (requires CDSE credentials)
        cdse_credentials: Optional dict with CDSE_CLIENT_ID, CDSE_CLIENT_SECRET and/or
            CDSE_USERNAME, CDSE_PASSWORD. Use when env vars are not visible (e.g. notebook config).
        s1_only: If True, fetch only Sentinel-1 from CDSE (no MODIS, no DEM, no Planetary Computer).

    Returns:
        Dict with keys: "s1", "modis", "dem", and "s1_source" ("copernicus" or "planetary_computer")
    """
    if s1_collections is None:
        s1_collections = ["sentinel-1-grd"]
    if use_copernicus_s1:
        s1_items = fetch_sentinel1_copernicus(
            bbox, flood_date, days_window=s1_days_window, cdse_credentials=cdse_credentials
        )
        s1_source = "copernicus"
        if not s1_items:
            logger.warning("No Sentinel-1 from Copernicus; S1 list will be empty (Copernicus only, no fallback).")
    else:
        s1_items = fetch_sentinel1(bbox, flood_date, days_window=s1_days_window, collections=s1_collections)
        s1_source = "planetary_computer"
    if s1_only:
        modis_items = []
        dem_items = []
    else:
        modis_items = fetch_modis(bbox, flood_date, days_back=modis_days_back)
        dem_items = fetch_dem(bbox)

    return {
        "s1": s1_items,
        "modis": modis_items,
        "dem": dem_items,
        "s1_source": s1_source,
    }

def download_and_load_assets(
    data_items: Dict[str, List[Any]],
    bbox: List[float],
    s1_preloaded: Optional[xr.DataArray] = None,
    cdse_credentials: Optional[Dict[str, str]] = None,
    s1_only: bool = False,
    s1_target_resolution_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Downloads/Streams the actual imagery from the cloud into Xarray DataArrays.
    
    This function takes STAC items from fetch_data_stac() and loads the actual
    raster data into xarray DataArrays. It uses stackstac for MODIS (handles
    temporal stacking automatically) and rioxarray for DEM and Sentinel-1.
    
    Args:
        data_items: Dictionary of STAC items from fetch_data_stac() with keys:
                   - "modis": List of MODIS STAC items
                   - "dem": List of DEM STAC items
                   - "s1": List of Sentinel-1 STAC items (can be empty if s1_preloaded is provided)
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
        s1_preloaded: Optional pre-loaded Sentinel-1 DataArray (e.g. from a manual get_s1_data
            call). When provided, used as "s1" output and the empty data_items["s1"] check is skipped.
        cdse_credentials: Optional CDSE credentials dict for Copernicus (used when loading S1
            from STAC items). Same format as fetch_data_stac(cdse_credentials=...).
        s1_only: If True, load only Sentinel-1 from CDSE (skip MODIS and DEM; no Planetary Computer required).
        s1_target_resolution_m: When set (e.g. 500), resample S1 during read to this resolution in meters.
            Reduces memory ~50x for large tiles (avoids loading full 10m native res before resampling).
    
    Returns:
        Dictionary containing xarray DataArrays:
            - "modis": MODIS time series stack (time, band, y, x)
            - "dem": DEM elevation data (y, x)
            - "s1": Sentinel-1 SAR data (y, x)
    
    Raises:
        ValueError: If required data items are missing (MODIS/DEM, or S1 when s1_preloaded not given)
        IndexError: If no items found in data_items lists
    
    Example:
        >>> data_items = fetch_data_stac(bbox, "2024-01-15", cdse_credentials=config_creds)
        >>> datasets = download_and_load_assets(data_items, bbox, cdse_credentials=config_creds)
        >>> # Or with manual S1: download_and_load_assets(data_items, bbox, s1_preloaded=s1_array)
    """
    # Validate input
    if not s1_only:
        if not data_items.get('modis'):
            raise ValueError("No MODIS items found. Cannot load MODIS data.")
        if not data_items.get('dem'):
            raise ValueError("No DEM items found. Cannot load DEM data.")
    if not data_items.get('s1') and s1_preloaded is None:
        raise ValueError(
            "No Sentinel-1 items found. Cannot load Sentinel-1 data. "
            "Either ensure CDSE credentials are set and fetch_data_stac returns S1 items, "
            "or pass s1_preloaded=your_s1_array from a manual fetch."
        )
    
    if s1_only:
        modis_ds = None
        dem_ds = None
    else:
        # 1. Load MODIS Stack (Time, Band, Y, X)
        if planetary_computer is None or stackstac is None:
            raise ImportError("s1_only=False requires planetary-computer and stackstac. pip install planetary-computer stackstac")
        logger.info(f"Loading {len(data_items['modis'])} MODIS scenes...")
        items_to_stack = []
        sign_failed = False
        for item in data_items['modis']:
            if sign_failed:
                items_to_stack.append(item)
                continue
            try:
                signed_item = planetary_computer.sign(item)
                items_to_stack.append(signed_item)
            except Exception as e:
                logger.warning(f"planetary_computer.sign failed for {item.id}: {e}")
                logger.info("Using items directly (from sign_inplace catalog).")
                sign_failed = True
                items_to_stack.append(item)
        if len(items_to_stack) == 0:
            raise ValueError("No MODIS items to load.")
        modis_ds = stackstac.stack(
            items_to_stack,
            assets=["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04",
                    "sur_refl_b05", "sur_refl_b06", "sur_refl_b07", "sur_refl_state_500m"],
            bounds=bbox,
            epsg=4326
        )
        if modis_ds.shape[-2] == 0 or modis_ds.shape[-1] == 0:
            raise ValueError(f"MODIS stack is empty for bbox {bbox}. Try a larger bbox or verify data coverage.")
        # 2. Load DEM
        logger.info("Loading DEM...")
        dem_item = data_items['dem'][0]
        dem_item_signed = planetary_computer.sign(dem_item)
        dem_asset_names = ["data", "elevation", "dem", "height"]
        dem_url = None
        for asset_name in dem_asset_names:
            if asset_name in dem_item_signed.assets:
                dem_url = dem_item_signed.assets[asset_name].href
                logger.info(f"Using DEM asset: {asset_name}")
                break
        if dem_url is None:
            available_assets = list(dem_item_signed.assets.keys())
            raise ValueError(f"No recognized DEM asset found. Available assets: {available_assets}")
        dem_ds = rioxarray.open_rasterio(dem_url)
        dem_ds = _clip_box_wgs84(dem_ds, bbox, label="DEM")

    # 3. Load Sentinel-1 (Label Reference)
    s1_list = data_items.get("s1") or []
    if s1_preloaded is not None:
        s1_ds = s1_preloaded
        logger.info("Using pre-loaded Sentinel-1 data (s1_preloaded).")
    elif not s1_list:
        raise ValueError(
            "No Sentinel-1 items to load. With Copernicus-only mode, ensure CDSE credentials are set "
            "and that Copernicus has S1 coverage for your bbox and date, or pass s1_preloaded=..."
        )
    else:
        s1_item = s1_list[0]
        s1_source = data_items.get("s1_source", "planetary_computer")
        cdse_token = _get_cdse_token(cdse_credentials) if s1_source == "copernicus" else None

        if s1_source == "copernicus":
            # Sentinel-1 from Copernicus only: require CDSE credentials; never use Planetary Computer/AWS.
            # CDSE STAC assets use S3-compatible storage at eodata.dataspace.copernicus.eu.
            # We need S3 keys (from https://eodata-s3keysmanager.dataspace.copernicus.eu/) so
            # GDAL uses CDSE's endpoint, not Amazon's.
            if not _configure_cdse_s3_env(cdse_credentials):
                raise ValueError(
                    "Sentinel-1 is from Copernicus but CDSE S3 credentials are missing. "
                    "CDSE assets use S3-compatible storage at eodata.dataspace.copernicus.eu (not AWS). "
                    "Add CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY from "
                    "https://eodata-s3keysmanager.dataspace.copernicus.eu/ to Colab Secrets or environment."
                )
            if not cdse_token:
                logger.warning(
                    "CDSE OAuth token not set; STAC search may have used cached token. "
                    "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET for search."
                )
            logger.info("Loading Sentinel-1 data from Copernicus CDSE (eodata.dataspace.copernicus.eu, no AWS)...")
            vv_href = s1_item.assets["vv"].href
            s1_ds = None
            try:
                s1_ds = _read_s1_vv_window_with_proj_metadata(
                    s1_item, bbox, label="Sentinel-1",
                    cdse_token=cdse_token, cdse_credentials=cdse_credentials,
                    target_resolution_m=s1_target_resolution_m,
                )
            except ValueError as e_proj:
                if "proj:epsg" not in str(e_proj) and "proj:transform" not in str(e_proj):
                    raise
                logger.info("Sentinel-1 STAC item has no proj:*; reading CRS/transform from file.")
                s1_ds = _read_s1_vv_window_from_file(
                    vv_href, bbox, label="Sentinel-1",
                    cdse_token=cdse_token, cdse_credentials=cdse_credentials,
                    target_resolution_m=s1_target_resolution_m,
                )
                if s1_ds is None:
                    logger.info("Sentinel-1 file has no CRS/transform; loading full raster and clipping to bbox.")
                    item_geom = getattr(s1_item, "geometry", None)
                    s1_ds = _read_s1_vv_full_then_clip(
                        vv_href, bbox, label="Sentinel-1", item_geometry=item_geom,
                        cdse_token=cdse_token, cdse_credentials=cdse_credentials,
                        target_resolution_m=s1_target_resolution_m,
                    )
            if s1_ds is None:
                raise ValueError("Sentinel-1 could not be loaded from Copernicus.")
        else:
            # Planetary Computer only when use_copernicus_s1=False (s1_source is planetary_computer).
            logger.info("Loading Sentinel-1 data from Planetary Computer (bbox-windowed via stackstac)...")
            s1_item_signed = planetary_computer.sign(s1_item)
            s1_ds = stackstac.stack(
                [s1_item_signed],
                assets=["vv"],
                bounds=bbox,
                epsg=4326,
                resolution=0.0001,
            )
            try:
                s1_ds = s1_ds.load()
            except AttributeError as e:
                if "to_epsg" not in str(e):
                    raise
                logger.info("stackstac failed opening Sentinel-1 (missing CRS); trying fallbacks.")
                vv_href = s1_item_signed.assets["vv"].href
                s1_ds = None
                try:
                    s1_ds = _read_s1_vv_window_with_proj_metadata(
                        s1_item_signed, bbox, label="Sentinel-1",
                        cdse_token=cdse_token, target_resolution_m=s1_target_resolution_m,
                    )
                except ValueError as e_proj:
                    if "proj:epsg" not in str(e_proj) and "proj:transform" not in str(e_proj):
                        raise
                    logger.info("Sentinel-1 STAC item has no proj:*; reading CRS/transform from file.")
                    s1_ds = _read_s1_vv_window_from_file(
                        vv_href, bbox, label="Sentinel-1",
                        cdse_token=cdse_token, target_resolution_m=s1_target_resolution_m,
                    )
                    if s1_ds is None:
                        logger.info("Sentinel-1 file has no CRS/transform; loading full raster and clipping to bbox.")
                        item_geom = getattr(s1_item_signed, "geometry", None) or getattr(s1_item, "geometry", None)
                        s1_ds = _read_s1_vv_full_then_clip(
                            vv_href, bbox, label="Sentinel-1", item_geometry=item_geom,
                            cdse_token=cdse_token, target_resolution_m=s1_target_resolution_m,
                        )
                if s1_ds is None:
                    raise ValueError("Sentinel-1 could not be loaded with any fallback.")

    # Validate Sentinel-1 window is non-empty
    if s1_ds.shape[-2] == 0 or s1_ds.shape[-1] == 0:
        raise ValueError(f"Sentinel-1 window is empty for bbox {bbox}. Try a larger bbox or different date.")

    return {
        "modis": modis_ds,
        "dem": dem_ds,
        "s1": s1_ds
    }

