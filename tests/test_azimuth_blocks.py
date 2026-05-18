"""Unit tests for azimuth blocking functionality."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from dolphin._types import Bbox
from dolphin.io import write_arr
from dolphin.workflows.config import DisplacementWorkflow

from disp_nisar._azimuth_blocks import (
    BlockWindow,
    FullFrameGrid,
    block_bounds,
    build_full_frame_grid,
    compute_block_windows,
    resolve_overlap,
)


class TestBlockWindow:
    """Test BlockWindow NamedTuple."""

    def test_block_window_creation(self):
        """Test that BlockWindow can be created with block_index field."""
        block = BlockWindow(
            block_index=0,
            read_start=0,
            read_stop=100,
            write_start=0,
            write_stop=90,
        )
        assert block.block_index == 0
        assert block.read_start == 0
        assert block.read_stop == 100
        assert block.write_start == 0
        assert block.write_stop == 90

    def test_block_window_properties(self):
        """Test BlockWindow computed properties."""
        block = BlockWindow(
            block_index=1,
            read_start=50,
            read_stop=150,
            write_start=60,
            write_stop=140,
        )
        assert block.read_height == 100
        assert block.write_height == 80
        assert block.write_offset_in_block == 10

    def test_block_index_no_conflict_with_tuple_method(self):
        """Test that block_index doesn't conflict with tuple.index() method."""
        block = BlockWindow(
            block_index=5,
            read_start=0,
            read_stop=100,
            write_start=0,
            write_stop=90,
        )
        # Should be able to access both the field and the tuple method
        assert block.block_index == 5
        # tuple.index() should still work to find elements
        assert block.index(5) == 0  # finds 5 (block_index value) at position 0


class TestComputeBlockWindows:
    """Test compute_block_windows function."""

    def test_single_block(self):
        """Test with single block covering entire frame."""
        windows = compute_block_windows(total_rows=500, num_blocks=1, overlap=10)
        assert len(windows) == 1
        assert windows[0].block_index == 0
        assert windows[0].read_start == 0
        assert windows[0].read_stop == 500
        assert windows[0].write_start == 0
        assert windows[0].write_stop == 500

    def test_multiple_blocks_with_overlap(self):
        """Test multiple blocks with halos."""
        windows = compute_block_windows(total_rows=500, num_blocks=5, overlap=7)
        assert len(windows) == 5

        # First block: no halo at top
        assert windows[0].read_start == 0
        assert windows[0].write_start == 0
        assert windows[0].read_stop > windows[0].write_stop  # has bottom halo

        # Middle blocks: halos on both sides
        for i in range(1, 4):
            assert windows[i].read_start < windows[i].write_start  # top halo
            assert windows[i].read_stop > windows[i].write_stop  # bottom halo

        # Last block: no halo at bottom
        assert windows[-1].write_stop == 500
        assert windows[-1].read_stop == 500
        assert windows[-1].read_start < windows[-1].write_start  # has top halo

    def test_blocks_are_contiguous(self):
        """Test that write windows are contiguous and cover entire frame."""
        total_rows = 1000
        windows = compute_block_windows(
            total_rows=total_rows, num_blocks=7, overlap=15
        )

        # Check first block starts at 0
        assert windows[0].write_start == 0

        # Check blocks are contiguous
        for i in range(len(windows) - 1):
            assert windows[i].write_stop == windows[i + 1].write_start

        # Check last block ends at total_rows
        assert windows[-1].write_stop == total_rows

    def test_zero_overlap(self):
        """Test blocks with no overlap."""
        windows = compute_block_windows(total_rows=400, num_blocks=4, overlap=0)
        assert len(windows) == 4

        for window in windows:
            # With no overlap, read and write windows should be identical
            assert window.read_start == window.write_start
            assert window.read_stop == window.write_stop

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="num_blocks must be >= 1"):
            compute_block_windows(total_rows=500, num_blocks=0, overlap=10)

        with pytest.raises(ValueError, match="total_rows.*must be >= num_blocks"):
            compute_block_windows(total_rows=5, num_blocks=10, overlap=10)

        with pytest.raises(ValueError, match="overlap must be >= 0"):
            compute_block_windows(total_rows=500, num_blocks=5, overlap=-1)


class TestBlockBounds:
    """Test block_bounds function."""

    def test_block_bounds_calculation(self):
        """Test conversion from block window to projected bounds."""
        # Create a frame grid
        frame = FullFrameGrid(
            bounds=Bbox(left=100.0, bottom=200.0, right=200.0, top=300.0),
            epsg=32610,
            x_res=10.0,
            y_res=10.0,
            rows=10,
            cols=10,
        )

        # Create a block covering first half of rows (0-5)
        block = BlockWindow(
            block_index=0,
            read_start=0,
            read_stop=5,
            write_start=0,
            write_stop=5,
        )

        bounds = block_bounds(frame, block)

        # X bounds should span full frame
        assert bounds.left == frame.bounds.left
        assert bounds.right == frame.bounds.right

        # Y bounds should cover top half (rows 0-5)
        # For north-up rasters, top is higher Y value
        assert bounds.top == frame.bounds.top
        # After 5 rows of 10m pixels, Y should decrease by 50m
        expected_bottom = frame.bounds.top - (5 * frame.y_res)
        assert abs(bounds.bottom - expected_bottom) < 1e-6

    def test_block_bounds_full_frame(self):
        """Test block covering entire frame."""
        frame = FullFrameGrid(
            bounds=Bbox(left=0.0, bottom=0.0, right=1000.0, top=1000.0),
            epsg=32610,
            x_res=5.0,
            y_res=5.0,
            rows=200,
            cols=200,
        )

        block = BlockWindow(
            block_index=0,
            read_start=0,
            read_stop=200,
            write_start=0,
            write_stop=200,
        )

        bounds = block_bounds(frame, block)

        # Should match frame bounds exactly
        assert bounds.left == frame.bounds.left
        assert bounds.right == frame.bounds.right
        assert bounds.top == frame.bounds.top
        assert bounds.bottom == frame.bounds.bottom


class TestResolveOverlap:
    """Test resolve_overlap function."""

    def test_resolve_overlap_uses_max(self):
        """Test that overlap uses max of half_window x and y."""
        mock_cfg = MagicMock()

        # Test with y larger
        mock_cfg.phase_linking.half_window.x = 5
        mock_cfg.phase_linking.half_window.y = 11
        assert resolve_overlap(mock_cfg) == 11

        # Test with x larger
        mock_cfg.phase_linking.half_window.x = 15
        mock_cfg.phase_linking.half_window.y = 8
        assert resolve_overlap(mock_cfg) == 15

        # Test with equal
        mock_cfg.phase_linking.half_window.x = 7
        mock_cfg.phase_linking.half_window.y = 7
        assert resolve_overlap(mock_cfg) == 7


class TestBuildFullFrameGrid:
    """Test build_full_frame_grid function."""

    def test_build_full_frame_grid(self):
        """Test building full frame grid from bounds."""
        mock_cfg = MagicMock()
        bounds = Bbox(left=100.0, bottom=200.0, right=300.0, top=400.0)
        mock_cfg.output_options.bounds = bounds

        x_res = 5.0
        y_res = 5.0
        epsg = 32610

        grid = build_full_frame_grid(mock_cfg, x_res, y_res, epsg)

        assert grid.bounds == bounds
        assert grid.epsg == epsg
        assert grid.x_res == x_res
        assert grid.y_res == y_res
        # Width = 200m, x_res = 5m -> 40 cols
        assert grid.cols == 40
        # Height = 200m, y_res = 5m -> 40 rows
        assert grid.rows == 40

    def test_build_full_frame_grid_no_bounds_raises(self):
        """Test that missing bounds raises error."""
        mock_cfg = MagicMock()
        mock_cfg.output_options.bounds = None

        with pytest.raises(ValueError, match="bounds must be set"):
            build_full_frame_grid(mock_cfg, 5.0, 5.0, 32610)

    def test_geotransform_property(self):
        """Test FullFrameGrid geotransform property."""
        grid = FullFrameGrid(
            bounds=Bbox(left=100.0, bottom=200.0, right=200.0, top=300.0),
            epsg=32610,
            x_res=10.0,
            y_res=10.0,
            rows=10,
            cols=10,
        )

        gt = grid.geotransform

        # Standard geotransform: (x_origin, x_pixel_size, x_rot,
        #                          y_origin, y_rot, -y_pixel_size)
        assert gt[0] == 100.0  # left
        assert gt[1] == 10.0  # x_res (positive)
        assert gt[2] == 0.0  # no rotation
        assert gt[3] == 300.0  # top
        assert gt[4] == 0.0  # no rotation
        assert gt[5] == -10.0  # -y_res (negative for north-up)


class TestStageInputsForBlock:
    """Test _stage_inputs_for_block function."""

    @patch("disp_nisar._azimuth_blocks._stage_input_to_local")
    def test_always_stages_all_files(self, mock_stage):
        """Test that all files are staged (the bug fix)."""
        from disp_nisar._azimuth_blocks import _stage_inputs_for_block

        # Create mock configuration
        mock_cfg = MagicMock()
        mock_cfg.cslc_file_list = [
            Path("/local/file1.h5"),
            Path("/vsis3/bucket/file2.h5"),
            Path("/local/file3.h5"),
        ]
        mock_cfg.input_options.subdataset = "/some/subdataset"

        # Create mock block and frame
        block = BlockWindow(
            block_index=0,
            read_start=0,
            read_stop=100,
            write_start=0,
            write_stop=90,
        )
        frame = FullFrameGrid(
            bounds=Bbox(left=0.0, bottom=0.0, right=1000.0, top=1000.0),
            epsg=32610,
            x_res=10.0,
            y_res=10.0,
            rows=100,
            cols=100,
        )

        staging_dir = Path("/tmp/staging")

        # Mock the staging function to return different paths
        mock_stage.side_effect = [
            Path("/tmp/staging/file1_block00.tif"),
            Path("/tmp/staging/file2_block00.tif"),
            Path("/tmp/staging/file3_block00.tif"),
        ]

        staged_files, new_subdataset = _stage_inputs_for_block(
            mock_cfg, block, staging_dir, frame
        )

        # Verify all files were staged (not just remote ones)
        assert mock_stage.call_count == 3
        assert len(staged_files) == 3

        # Verify subdataset is set to None (all files are GTiffs now)
        assert new_subdataset is None

        # Verify each file was staged with correct parameters
        for i, call in enumerate(mock_stage.call_args_list):
            args, kwargs = call
            assert str(args[0]) == str(mock_cfg.cslc_file_list[i])
            assert args[1] == "/some/subdataset"
            assert args[2] == block
            assert args[3] == staging_dir
            assert args[4] == frame


class TestMaskCropping:
    """Test mask cropping to block dimensions."""

    @pytest.fixture
    def temp_raster(self, tmp_path):
        """Create a temporary test raster."""
        raster_path = tmp_path / "test_raster.tif"
        # Create a 100x100 raster
        data = np.ones((100, 100), dtype=np.uint8)
        # Mark edges as nodata (0)
        data[0:10, :] = 0  # top
        data[-10:, :] = 0  # bottom
        data[:, 0:10] = 0  # left
        data[:, -10:] = 0  # right

        # Create a simple geotransform
        geotransform = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
        projection = "EPSG:32610"

        write_arr(
            arr=data,
            output_name=raster_path,
            geotransform=geotransform,
            projection=projection,
        )
        return raster_path

    @patch("disp_nisar._azimuth_blocks.io.load_gdal")
    @patch("disp_nisar._azimuth_blocks.io.write_arr")
    def test_crop_frame_mask_to_block(self, mock_write, mock_load, temp_raster):
        """Test that mask is cropped correctly to block size."""
        from disp_nisar._azimuth_blocks import _crop_frame_mask_to_block

        # Mock load_gdal to return a subset of data
        mock_data = np.ones((50, 100), dtype=np.uint8)
        mock_load.return_value = mock_data

        block = BlockWindow(
            block_index=0,
            read_start=0,
            read_stop=50,
            write_start=0,
            write_stop=40,
        )

        frame = FullFrameGrid(
            bounds=Bbox(left=0.0, bottom=0.0, right=100.0, top=100.0),
            epsg=32610,
            x_res=1.0,
            y_res=1.0,
            rows=100,
            cols=100,
        )

        output_path = Path("/tmp/cropped_mask.tif")

        _crop_frame_mask_to_block(
            frame_mask=temp_raster,
            template=temp_raster,
            block=block,
            out_path=output_path,
            frame=frame,
        )

        # Verify load_gdal was called with correct row slice
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["rows"] == slice(0, 50)
        assert call_kwargs["cols"] == slice(None)

        # Verify write_arr was called
        assert mock_write.call_count == 1

        # Verify the output has correct dimensions
        write_call_kwargs = mock_write.call_args[1]
        written_arr = write_call_kwargs["arr"]
        assert written_arr.shape == (50, 100)


class TestIntegration:
    """Integration tests for azimuth blocking workflow."""

    def test_blocks_cover_entire_frame_no_gaps_overlaps(self):
        """Test that blocks cover entire frame with no gaps or overlaps."""
        total_rows = 1234
        num_blocks = 8
        overlap = 20

        windows = compute_block_windows(total_rows, num_blocks, overlap)

        # Collect all rows that will be written
        written_rows = set()
        for window in windows:
            for row in range(window.write_start, window.write_stop):
                assert row not in written_rows, f"Row {row} written by multiple blocks"
                written_rows.add(row)

        # Check all rows are covered
        assert written_rows == set(range(total_rows))

    def test_read_windows_include_halos(self):
        """Test that read windows properly include halos for processing."""
        windows = compute_block_windows(total_rows=500, num_blocks=5, overlap=15)

        for i, window in enumerate(windows):
            write_height = window.write_height
            read_height = window.read_height

            if i == 0:
                # First block: halo only on bottom
                assert read_height >= write_height
            elif i == len(windows) - 1:
                # Last block: halo only on top
                assert read_height >= write_height
            else:
                # Middle blocks: halos on both sides
                assert read_height >= write_height + 2 * 15  # overlap on each side
