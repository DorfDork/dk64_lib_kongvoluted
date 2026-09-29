from __future__ import annotations

import copy
import math

from dataclasses import dataclass, field, replace
from tempfile import TemporaryFile
from typing import Callable

from dk64_lib.f3dex2.vertex import Vertex
from dk64_lib.f3dex2.triangle import Triangle

from dk64_lib.binary_reader import BinaryReader
from dk64_lib.f3dex2 import commands
from dk64_lib.f3dex2.commands import get_command, DL_Command

from dk64_lib.file_io import get_bytes


@dataclass(frozen=True, slots=True)
class DisplayListExpansion:
    """
        This section of the geometry is currently not fully understood
        As such, there are signficant gaps in the known values
        
        Currently, the only known information is that the third u32 points
        to a display list. This display list uses the entire vertex data
        when calling VTX commands instead of a segmented chunk
    """
    unknown_1: int
    unknown_2: int
    display_list_offset: int
    unknown_4: int

    @classmethod
    def from_bytes(cls, raw_data: bytes) -> "DisplayListExpansion":
        reader = BinaryReader(raw_data)
        return cls(
            unknown_1=reader.read_u32(0),
            unknown_2=reader.read_u32(4),
            display_list_offset=reader.read_u32(8),
            unknown_4=reader.read_u32(12),
        )


@dataclass(frozen=True, slots=True)
class DisplayListChunkData:
    r: int
    g: int
    b: int
    unknown_char: int
    mips_instruction: bytes
    unknown_flag: int
    dl_1_start: int
    dl_1_size: int
    dl_2_start: int
    dl_2_size: int
    dl_3_start: int
    dl_3_size: int
    dl_4_start: int
    dl_4_size: int
    vertex_start: int
    vertex_size: int

    @classmethod
    def from_bytes(cls, raw_data: bytes) -> "DisplayListChunkData":
        reader = BinaryReader(raw_data)
        return cls(
            r=reader.read_u8(0),
            g=reader.read_u8(1),
            b=reader.read_u8(2),
            unknown_char=reader.read_u8(3),
            mips_instruction=reader.read_at(4, 4),
            unknown_flag=reader.read_u32(8),
            dl_1_start=reader.read_u32(12),
            dl_1_size=reader.read_u32(16),
            dl_2_start=reader.read_u32(20),
            dl_2_size=reader.read_u32(24),
            dl_3_start=reader.read_u32(28),
            dl_3_size=reader.read_u32(32),
            dl_4_start=reader.read_u32(36),
            dl_4_size=reader.read_u32(40),
            vertex_start=reader.read_u32(44),
            vertex_size=reader.read_u32(48),
        )

    @property
    def vertex_start_size(self) -> dict[int, tuple[int, int]]:
        return {
            self.dl_1_start: (self.vertex_start, self.vertex_size),
            self.dl_2_start: (self.vertex_start, self.vertex_size),
            self.dl_3_start: (self.vertex_start, self.vertex_size),
            self.dl_4_start: (self.vertex_start, self.vertex_size),
        }


class DisplayList:
    def __init__(
        self,
        raw_data: bytes,
        raw_vertex_data: bytes,
        vertex_pointer: int,
        offset: int,
        branches: list["DisplayList"] = None,
        branched: bool = False,
    ):
        """An object representation of the N64 display list

        Args:
            raw_data (bytes): Raw data of the display list
            raw_vertex_data (bytes): Raw data of the verticies associated with this display list
            vertex_pointer (int): The display list's pointer in the vertex data
            offset (int): Localized offset of this display list
            branches (list[DisplayList], optional): The display list's branches. Defaults to None.
            branched (bool, optional): Whether the display list is branched or not. Defaults to False.
        """
        
        
        self._raw_data = raw_data
        self.branches = branches if branches else list()
        self.raw_vertex_data = raw_vertex_data
        self.vertex_pointer = vertex_pointer
        self.offset = offset
        self.is_branched = branched

    def __repr__(self):
        if self.branches:
            return f"DisplayList({self.offset=}, {self.size=}, {self.num_commands=}, branches={len(self.branches)})"
        return f"DisplayList({self.offset=}, {self.size=}, {self.num_commands=})"

    def __eq__(self, obj):
        if isinstance(obj, int):
            return self.offset == obj
        return self == obj

    @property
    def raw_vertex_data(self):
        return self._raw_vertex_data

    @raw_vertex_data.setter
    def raw_vertex_data(self, raw_vertex_data):
        self._raw_vertex_data = raw_vertex_data
        for branch in self.branches:
            branch.raw_vertex_data = raw_vertex_data

    @property
    def size(self) -> int:
        """Returns size of display list raw data

        Returns:
            int: Size of display list raw data
        """
        return len(self._raw_data)

    @property
    def num_commands(self) -> int:
        """Returns number of commands in the display list

        Returns:
            int: Number of commands
        """
        return int(self.size / 8)

    @property
    def vertex_buffers(self) -> list[commands.G_VTX]:
        """Returns a list of G_VTX objects

        Returns:
            list[commands.G_VTX]: A list of vertex buffer objects
        """
        return [cmd for cmd in self.commands if cmd.opcode == b"\x01"]

    @property
    def vertex_count(self) -> int:
        return sum([vtx.vertex_count for vtx in self.vertex_buffers])

    @property
    def recursive_vertex_count(self) -> int:
        total_verticies = 0
        for branch_dl in self.branches:
            total_verticies += branch_dl.recursive_vertex_count
        return self.vertex_count + total_verticies

    @property
    def triangles(self) -> list[list[Triangle]]:
        """Returns a 2d list of triangle data in the display list, each sub-list corresponding to an adjacent vertex group

        Returns:
            list[list[commands.G_TRI1]]: A 2d list of triangle data
        """
        ret_list = list()
        tri_list = list()
        for cmd in self.commands:

            # Read vertex buffer and create new triangle list
            if cmd.opcode == b"\x01":
                tri_list = list()
                ret_list.append(tri_list)
                continue

            # Read triangle data and add it to the triangle list
            if cmd.opcode == b"\x05":
                tri = Triangle.from_tri1(cmd)
                tri_list.append(tri)
                continue

            # Read dual triangle data and add it to the triangle list
            if cmd.opcode == b"\x06":
                tri1, tri2 = Triangle.from_tri2(cmd)
                tri_list.append(tri1)
                tri_list.append(tri2)
                continue

            if cmd.opcode == b"\xDE":
                branched_dl = self.get_branch_by_offset(
                    int.from_bytes(cmd.address, "big")
                )
                ret_list.extend(branched_dl.triangles)
                continue

        return ret_list

    @property
    def verticies(self) -> list[list[Vertex]]:
        """Returns a 2d list of Vertex, each sub-list corresponding to an adjacent triangle group

        Returns:
            list[list[Vertex]]: A 2d list of vertex data
        """
        ret_list = list()
        vert_list = list()

        for cmd in self.commands:

            if cmd.opcode == b"\x01":
                vertex_buffer_start = self.vertex_pointer + int.from_bytes(
                    cmd.address, "big"
                )
                vertex_buffer_end = vertex_buffer_start + cmd.vertex_count * 16
                vertex_data = self._raw_vertex_data[
                    vertex_buffer_start:vertex_buffer_end
                ]

                if vertex_buffer_end > len(self._raw_vertex_data):
                    vertex_buffer_start = int.from_bytes(cmd.address, "big")
                    vertex_buffer_end = vertex_buffer_start + cmd.vertex_count * 16
                    vertex_data = self._raw_vertex_data[
                        vertex_buffer_start:vertex_buffer_end
                    ]

                # Vertex data is 16 bytes long
                vert_start = 0
                vert_end = vert_start + 16

                for _ in range(cmd.vertex_count):
                    # Read the raw data and create a Vertex object out of it
                    vert_list.append(Vertex.from_bytes(vertex_data[vert_start:vert_end]))

                    # Move the vertex start and end 16 bytes ahead
                    vert_start = vert_end
                    vert_end = vert_start + 16

                # Append the vert list and reset
                ret_list.append(vert_list)
                vert_list = list()
                continue

            if cmd.opcode == b"\xDE":
                branched_dl = self.get_branch_by_offset(
                    int.from_bytes(cmd.address, "big")
                )
                ret_list.extend(branched_dl.verticies)
                continue

        return ret_list

    @property
    def commands(self) -> list[DL_Command]:
        """Returns the F3DEX2 commands in the display list

        Returns:
            list[DL_Command]: A list of F3DEX2 commands
        """
        ret_list = list()
        for command_pos in range(self.num_commands):
            command_bytes = self._raw_data[command_pos * 8 : command_pos * 8 + 8]
            # Parse each raw command in an object for easy parsing of data
            if command := get_command(command_bytes):
                ret_list.append(command)
        return ret_list

    def get_branch_by_offset(self, offset):
        if offset in self.branches:
            return self.branches[self.branches.index(offset)]
        return None


def create_display_lists(
    display_list_data: bytes,
    vertex_data: bytes,
    display_list_chunk_data: list[DisplayListChunkData],
    expansions: list[DisplayListExpansion] = None,
) -> list[DisplayList]:
    """Create a geometry's display lists given the display list data, vertex data, and display list chunk data

    Args:
        display_list_data (bytes): Display list data
        vertex_data (bytes): Vertex data
        display_list_chunk_data (list[DisplayListChunkData]): List of DisplayListChunkData
        expansions (list[DisplayListExpansion], optional): Any display list expansion data that might exist in the geometry file. Defaults to None.

    Returns:
        list[DisplayList]: A list of DisplayList objects
    """

    def read_display_lists(
        dl_pointer: int = 0, branched: bool = False, inherited_vertex_data: bytes = None
    ) -> list[DisplayList]:
        """Recursively read display lists

        Args:
            _dl_pointer (int, optional): The display list pointer. Defaults to 0.
            _branched (bool, optional): When the display list is branched or not. Defaults to False.
            _vertex_data (bytes, optional): Vertex data to override the standard vertex_data with. Defaults to None.


        Returns:
            list[DisplayList]: A list of DisplayList objects
        """
        nonlocal display_list_data, vertex_data, display_list_chunk_data, expansions
        ret_list = list()
        branches = list()

        # Generate a dict where the key is the dl offset and the value is a tuple containing the start and size of the vertices
        dl_vertex_starts = {
            k: v
            for chunk in display_list_chunk_data
            for k, v in chunk.vertex_start_size.items()
        }

        # Generate a list of offsets inlucded in the expansion data. These display lists will use the entire vertex data instead of a segment
        expansion_offsets = (
            [expansion.display_list_offset for expansion in expansions]
            if expansions
            else []
        )

        # Write the raw data to a temporary file so we can seek and read as necessary
        with TemporaryFile() as data_file:

            data_file.write(display_list_data)
            data_file.seek(dl_pointer)

            raw_data = b""
            vertex_pointer = 0
            old_vertex_start = 0

            branched_dls = dict()

            dl_raw_vertex_data = inherited_vertex_data or vertex_data

            # While we haven't reached the vertex start point, read each 8 byte command
            while command_bytes := get_bytes(data_file, 8):

                # Get the F3DEX2 command class and intantiate it as an object
                cmd = get_command(command_bytes)

                # Write the command bytes. This will become our DisplayList's _raw_data
                raw_data += command_bytes

                if cmd is None:
                    continue

                # * Handle branching display lists
                if cmd.opcode == b"\xDE":

                    branches.extend(
                        read_display_lists(
                            dl_pointer=int.from_bytes(cmd.address, "big"),
                            branched=True,
                            inherited_vertex_data=dl_raw_vertex_data,
                        )
                    )
                    continue

                # * Once we reach the end of the Display List, create the object and start fresh
                if cmd.opcode == b"\xDF":

                    # TODO: This is a relic of poor understanding of display list vertex data
                    # TODO: There has to be a cleaner method of processing this
                    if dl_vertex_starts.get(dl_pointer):
                        vertex_start, vertex_size = dl_vertex_starts[dl_pointer]
                        if vertex_start != old_vertex_start:
                            vertex_pointer = 0
                            old_vertex_start = vertex_start
                        dl_raw_vertex_data = vertex_data[
                            vertex_start : vertex_start + vertex_size
                        ]

                    # If the display list exists in the expansion array, then it uses the entire vertex data
                    if dl_pointer in expansion_offsets:
                        dl_raw_vertex_data = vertex_data
                        vertex_pointer = 0

                    # Check and see if the display list currently exists in the branches, if it does, add that instead
                    # Otherwise, generate a new one
                    display_list = branched_dls.get(dl_pointer)
                    if not display_list:
                        display_list = DisplayList(
                            raw_data=raw_data,
                            raw_vertex_data=dl_raw_vertex_data,
                            offset=dl_pointer,
                            vertex_pointer=vertex_pointer,
                            branches=branches,
                            branched=branched,
                        )

                    # Update relevant variables
                    ret_list.append(display_list)
                    branched_dls.update({dl.offset: dl for dl in display_list.branches})
                    vertex_pointer += display_list.vertex_count * 16

                    # Update the branches to use the parent vertex data
                    for branch in branches:
                        branch._raw_vertex_data = dl_raw_vertex_data

                    # Reset for the next display list
                    raw_data = b""
                    branches = list()
                    dl_pointer = data_file.tell()

                    # If this is a branched display list, break and return
                    if branched:
                        break
                    continue

        return ret_list

    return read_display_lists()

G_CULL_FRONT = 0x00000200
G_CULL_BACK = 0x00000400
G_FOG = 0x00010000
G_LIGHTING = 0x00020000
G_TEXTURE_GEN = 0x00040000
G_TEXTURE_GEN_LINEAR = 0x00080000

HILITE_SEGMENT = 0x05

CYCLE_TYPE_SHIFT = 20
CYCLE_TYPE_LENGTH = 2
G_CYC_2CYCLE = 1

CM_MIRROR = 0x1
CM_CLAMP = 0x2

_COLOR_A = ("COMBINED", "TEXEL0", "TEXEL1", "PRIMITIVE", "SHADE", "ENVIRONMENT", "1", "NOISE")
_COLOR_B = ("COMBINED", "TEXEL0", "TEXEL1", "PRIMITIVE", "SHADE", "ENVIRONMENT", "CENTER", "K4")
_COLOR_C = (
    "COMBINED", "TEXEL0", "TEXEL1", "PRIMITIVE", "SHADE", "ENVIRONMENT", "SCALE",
    "COMBINED_ALPHA", "TEXEL0_ALPHA", "TEXEL1_ALPHA", "PRIMITIVE_ALPHA", "SHADE_ALPHA",
    "ENV_ALPHA", "LOD_FRACTION", "PRIM_LOD_FRAC", "K5",
)
_COLOR_D = ("COMBINED", "TEXEL0", "TEXEL1", "PRIMITIVE", "SHADE", "ENVIRONMENT", "1")
_ALPHA_ABD = ("COMBINED", "TEXEL0", "TEXEL1", "PRIMITIVE", "SHADE", "ENVIRONMENT", "1")
_ALPHA_C = (
    "LOD_FRACTION", "TEXEL0", "TEXEL1", "PRIMITIVE", "SHADE", "ENVIRONMENT", "PRIM_LOD_FRAC"
)

CYCLE_FIELDS = ("A", "B", "C", "D", "A_alpha", "B_alpha", "C_alpha", "D_alpha")

DEFAULT_CYCLE = ("TEXEL0", "0", "SHADE", "0", "TEXEL0", "0", "SHADE", "0")
WHITE = (0xFF, 0xFF, 0xFF, 0xFF)


def _mux(table: tuple[str, ...], value: int) -> str:
    return table[value] if value < len(table) else "0"


@dataclass(frozen=True, slots=True)
class RenderState:
    combiner: tuple[tuple[str, ...], tuple[str, ...]] = (DEFAULT_CYCLE, DEFAULT_CYCLE)
    prim_color: tuple[int, int, int, int] = WHITE
    env_color: tuple[int, int, int, int] = WHITE
    lighting: bool = False
    cull_front: bool = False
    cull_back: bool = False
    fog: bool = False
    texture_gen: bool = False
    texture_gen_linear: bool = False
    hilite: bool = False
    two_cycle: bool = False
    mirror_s: bool = False
    mirror_t: bool = False
    mask_s: int = 0
    mask_t: int = 0
    shift_s: int = 0
    shift_t: int = 0
    other_mode_l: int = 0

    @property
    def alpha_reads_texture(self) -> bool:
        cycles = self.combiner[: 2 if self.two_cycle else 1]
        alpha_inputs = CYCLE_FIELDS.index("A_alpha")
        return any(
            name.startswith("TEXEL") for cycle in cycles for name in cycle[alpha_inputs:]
        )


def decode_combiner(
    command: commands.G_SETCOMBINE | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if command is None:
        return (DEFAULT_CYCLE, DEFAULT_CYCLE)
    return tuple(
        (
            _mux(_COLOR_A, getattr(command, f"color_a_{cycle}")),
            _mux(_COLOR_B, getattr(command, f"color_b_{cycle}")),
            _mux(_COLOR_C, getattr(command, f"color_c_{cycle}")),
            _mux(_COLOR_D, getattr(command, f"color_d_{cycle}")),
            _mux(_ALPHA_ABD, getattr(command, f"alpha_a_{cycle}")),
            _mux(_ALPHA_ABD, getattr(command, f"alpha_b_{cycle}")),
            _mux(_ALPHA_C, getattr(command, f"alpha_c_{cycle}")),
            _mux(_ALPHA_ABD, getattr(command, f"alpha_d_{cycle}")),
        )
        for cycle in (0, 1)
    )


def cycle_type_is_two(other_mode_h_data: int) -> bool:
    cycle_mask = (1 << CYCLE_TYPE_LENGTH) - 1
    return (other_mode_h_data >> CYCLE_TYPE_SHIFT) & cycle_mask == G_CYC_2CYCLE


VERTEX_SIZE = 0x10


@dataclass(frozen=True, slots=True)
class _TextureKey:
    image_index: int
    palette_index: int | None
    fmt: int
    size: int
    width: int
    height: int
    clamp_s: bool = False
    clamp_t: bool = False
    from_fallback_table: bool = False

    @property
    def material_name(self) -> str:
        palette = "none" if self.palette_index is None else str(self.palette_index)
        name = (
            f"tex_{self.image_index}_pal_{palette}_"
            f"f{self.fmt}_s{self.size}_{self.width}x{self.height}"
        )
        if self.clamp_s and self.clamp_t:
            return f"{name}_clamp_st"
        if self.clamp_s:
            return f"{name}_clamp_s"
        if self.clamp_t:
            return f"{name}_clamp_t"
        return name

    @property
    def image_filename(self) -> str:
        return f"{self.material_name}.png"

    @property
    def byte_size(self) -> int:
        return self.width * self.height * (4 << self.size) // 8


@dataclass(frozen=True, slots=True)
class _MeshGroup:
    vertices: tuple[Vertex, ...]
    triangles: tuple[Triangle, ...]
    texture: _TextureKey | None
    display_list_offset: int
    translucent: bool = False
    double_sided: bool = True
    vertex_bones: tuple[int, ...] = tuple()
    vertex_normals: tuple[tuple[float, float, float], ...] = tuple()
    render: RenderState | None = None
    optional_segment: int | None = None


@dataclass(frozen=True, slots=True)
class ModelBone:
    index: int
    parent: int
    local: tuple[float, float, float]
    world: tuple[float, float, float]
    channel: int = 0


@dataclass(frozen=True, slots=True)
class _ImageSource:
    index: int
    fmt: int
    size: int


@dataclass(frozen=True, slots=True)
class _TileDescriptor:
    fmt: int
    size: int
    line: int
    tmem: int
    palette: int
    clamp_s: bool
    clamp_t: bool
    mirror_s: bool = False
    mirror_t: bool = False
    mask_s: int = 0
    mask_t: int = 0
    shift_s: int = 0
    shift_t: int = 0


class _TextureState:
    def __init__(self, prefer_loaded_dimensions: bool = False):
        self._prefer_loaded_dimensions = prefer_loaded_dimensions
        self._pending_image: _ImageSource | None = None
        self._loaded_images: dict[int, _ImageSource] = {}
        self._tile_descriptors: dict[int, _TileDescriptor] = {}
        self._tile_rects: dict[int, commands.G_SETTILESIZE] = {}
        self._loaded_bytes: dict[int, int] = {}
        self._last_loaded_tile: int | None = None
        self._last_palette: _ImageSource | None = None
        self._active_tile: int | None = None
        self._texture_tile = 0
        self._texture_scale = (1.0, 1.0)
        self._combine: commands.G_SETCOMBINE | None = None
        self._prim_color = WHITE
        self._env_color = WHITE
        self._geometry_mode = 0
        self._two_cycle = False
        self._other_mode_l = 0
        self._hilite = False

    def clone(self) -> _TextureState:
        state = copy.copy(self)
        state._loaded_images = dict(self._loaded_images)
        state._tile_descriptors = dict(self._tile_descriptors)
        state._tile_rects = dict(self._tile_rects)
        state._loaded_bytes = dict(self._loaded_bytes)
        return state

    def apply(self, command: commands.DL_Command) -> None:
        if isinstance(command, commands.G_SETTIMG):
            self._pending_image = _ImageSource(
                index=command.address,
                fmt=command.fmt,
                size=command.size,
            )
            return

        if isinstance(command, (commands.G_LOADBLOCK, commands.G_LOADTILE)):
            loading = self._tile_descriptors.get(command.tile)
            loading_bits = 4 << loading.size if loading else 16
            if isinstance(command, commands.G_LOADBLOCK):
                texels = command.texel_count
            else:
                texels = (abs(command.lrs - command.uls) // 4 + 1) * (
                    abs(command.lrt - command.ult) // 4 + 1
                )
            self._loaded_bytes[command.tile] = texels * loading_bits // 8
            if self._pending_image is not None:
                self._loaded_images[command.tile] = self._pending_image
                self._last_loaded_tile = command.tile
            return

        if isinstance(command, commands.G_LOADTLUT):
            if self._pending_image is not None:
                self._last_palette = self._pending_image
            return

        if isinstance(command, commands.G_SETTILE):
            self._tile_descriptors[command.tile] = _TileDescriptor(
                fmt=command.fmt,
                size=command.size,
                line=command.line,
                tmem=command.tmem,
                palette=command.palette,
                clamp_s=bool(command.cm_s & CM_CLAMP),
                clamp_t=bool(command.cm_t & CM_CLAMP),
                mirror_s=bool(command.cm_s & CM_MIRROR),
                mirror_t=bool(command.cm_t & CM_MIRROR),
                mask_s=command.mask_s,
                mask_t=command.mask_t,
                shift_s=command.shift_s,
                shift_t=command.shift_t,
            )
            return

        if isinstance(command, commands.G_SETTILESIZE):
            if self._active_tile is None:
                self._active_tile = command.tile
            self._tile_rects[command.tile] = command
            self._hilite = False
            return

        if isinstance(command, commands.G_DL):
            if command.segment[0] == HILITE_SEGMENT and command.store_return_address:
                self._hilite = True
            return

        if isinstance(command, commands.G_TEXTURE):
            self._active_tile = command.tile if command.on else None
            self._texture_tile = command.tile
            self._texture_scale = tuple(
                1.0 if scale == 0xFFFF else scale / 0x10000
                for scale in (command.scale_s, command.scale_t)
            )
            return

        if isinstance(command, commands.G_SETCOMBINE):
            self._combine = command
            return

        if isinstance(command, commands.G_SETPRIMCOLOR):
            self._prim_color = (command.r, command.g, command.b, command.a)
            return

        if isinstance(command, commands.G_SETENVCOLOR):
            self._env_color = (command.r, command.g, command.b, command.a)
            return

        if isinstance(command, commands.G_GEOMETRYMODE):
            self._geometry_mode = (
                self._geometry_mode & ~command.clear_bits
            ) | command.set_bits
            return

        if isinstance(command, commands.G_SetOtherMode_H):
            if command.shift == CYCLE_TYPE_SHIFT:
                self._two_cycle = cycle_type_is_two(command.data)
            return

        if isinstance(command, commands.G_SetOtherMode_L):
            mask = ((1 << command.length) - 1) << command.shift
            self._other_mode_l = (self._other_mode_l & ~mask) | (command.data & mask)

    @property
    def geometry_mode(self) -> int:
        return self._geometry_mode

    @property
    def other_mode_l(self) -> int:
        return self._other_mode_l

    @property
    def texture_tile(self) -> int:
        return self._texture_tile

    @property
    def texture_scale(self) -> tuple[float, float]:
        return self._texture_scale

    @property
    def texture_tile_shift(self) -> tuple[int, int]:
        descriptor = self._tile_descriptors.get(self._texture_tile)
        return (descriptor.shift_s, descriptor.shift_t) if descriptor else (0, 0)

    @property
    def texture_tile_origin(self) -> tuple[int, int]:
        rect = self._tile_rects.get(self._texture_tile)
        return (rect.uls, rect.ult) if rect else (0, 0)

    @property
    def combiner_reads_texture(self) -> bool:
        cycles = decode_combiner(self._combine)[: 2 if self._two_cycle else 1]
        return any(name.startswith("TEXEL") for cycle in cycles for name in cycle)

    @property
    def render_state(self) -> RenderState:
        tile = self._tile_descriptors.get(self._active_tile)
        return RenderState(
            combiner=decode_combiner(self._combine),
            prim_color=self._prim_color,
            env_color=self._env_color,
            lighting=bool(self._geometry_mode & G_LIGHTING),
            cull_front=bool(self._geometry_mode & G_CULL_FRONT),
            cull_back=bool(self._geometry_mode & G_CULL_BACK),
            fog=bool(self._geometry_mode & G_FOG),
            texture_gen=bool(self._geometry_mode & G_TEXTURE_GEN),
            texture_gen_linear=bool(
                self._geometry_mode & G_TEXTURE_GEN
                and self._geometry_mode & G_TEXTURE_GEN_LINEAR
            ),
            hilite=self._hilite,
            two_cycle=self._two_cycle,
            mirror_s=tile.mirror_s if tile else False,
            mirror_t=tile.mirror_t if tile else False,
            mask_s=tile.mask_s if tile else 0,
            mask_t=tile.mask_t if tile else 0,
            shift_s=tile.shift_s if tile else 0,
            shift_t=tile.shift_t if tile else 0,
            other_mode_l=self._other_mode_l,
        )

    @property
    def active_texture(self) -> _TextureKey | None:
        if self._active_tile is None:
            return None

        descriptor = self._tile_descriptors.get(self._active_tile)
        if descriptor is None:
            return None

        source = self._loaded_images.get(self._active_tile)
        if source is None and self._last_loaded_tile is not None:
            source = self._loaded_images.get(self._last_loaded_tile)
        if source is None:
            return None

        tile_size = self._tile_size(self._active_tile)
        loaded_size = self._loaded_tile_dimensions(descriptor)
        if self._prefer_loaded_dimensions:
            dimensions = loaded_size or tile_size
        else:
            dimensions = tile_size or loaded_size
        if dimensions is None:
            return None

        palette_index = None
        if descriptor.fmt == 2 and self._last_palette is not None:
            palette_index = self._last_palette.index

        return _TextureKey(
            image_index=source.index,
            palette_index=palette_index,
            fmt=descriptor.fmt,
            size=descriptor.size,
            width=dimensions[0],
            height=dimensions[1],
            clamp_s=descriptor.clamp_s,
            clamp_t=descriptor.clamp_t,
        )

    def _tile_size(self, tile: int) -> tuple[int, int] | None:
        rect = self._tile_rects.get(tile)
        if rect is None:
            return None
        return (
            max(1, abs(rect.lrs - rect.uls) // 4 + 1),
            max(1, abs(rect.lrt - rect.ult) // 4 + 1),
        )

    def _loaded_tile_dimensions(
        self,
        descriptor: _TileDescriptor,
    ) -> tuple[int, int] | None:
        byte_count = self._loaded_bytes.get(self._last_loaded_tile)
        bits = 4 << descriptor.size
        width = descriptor.line * (4 if descriptor.size == 3 else 64 // bits)
        tile_size = self._tile_size(self._active_tile)
        if width <= 0 and tile_size is not None:
            width = tile_size[0]
        if not byte_count or width <= 0:
            return None

        height = max(1, byte_count * 8 // (width * bits))
        rect = self._tile_rects.get(self._active_tile)
        if rect is not None and rect.lrt:
            height = min(tile_size[1], height)
        return width, height


def _signed_16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


# Actor and prop model files (pointer tables 5 and 4) are relocatable blobs that reach their
# vertices, display lists and textures through raw pointers and segment addresses.
# ModelDecoder walks them like the RSP, tracking the vertex cache and bone matrix, and
# produces the same mesh groups as map geometry.

COMMAND_SIZE = 8
BONE_RECORD_SIZE = 0x10
MATRIX_SIZE = 0x40
VERTEX_CACHE_SIZE = 32
MAX_BRANCH_DEPTH = 64
NO_PARENT = 0xFF
BONE_MATRIX_SEGMENT = 0x04
G_RM_FORCE_BL = 0x4000


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    offset: int
    bone: int
    scale_s: float
    scale_t: float


@dataclass
class _MaterialMesh:
    texture: _TextureKey | None
    display_list_offset: int
    translucent: bool = False
    double_sided: bool = True
    render: RenderState | None = None
    optional_segment: int | None = None
    vertices: list[Vertex] = field(default_factory=list)
    vertex_bones: list[int] = field(default_factory=list)
    vertex_normals: list[tuple[float, float, float]] = field(default_factory=list)
    triangles: list[Triangle] = field(default_factory=list)
    indices: dict[tuple, int] = field(default_factory=dict)


def shift_scale(shift: int) -> float:
    if shift == 0:
        return 1.0
    if shift <= 10:
        return 1 / (1 << shift)
    return float(1 << (16 - shift))


def read_bones(
    raw_data: bytes,
    start: int,
    count: int,
) -> list[ModelBone]:
    if start < 0 or start + count * BONE_RECORD_SIZE > len(raw_data):
        raise ValueError("Bone table extends past the end of the file")

    reader = BinaryReader(raw_data)
    parents = list()
    channels = list()
    translations = list()
    for bone_index in range(count):
        offset = start + bone_index * BONE_RECORD_SIZE
        parents.append(reader.read_u8(offset))
        channels.append(reader.read_u8(offset + 2))
        translations.append(
            tuple(reader.read_f32(offset + 4 + axis * 4) for axis in range(3))
        )

    def world(bone_index: int) -> tuple[float, float, float]:
        origin = [0.0, 0.0, 0.0]
        visited = set()
        while bone_index not in visited:
            visited.add(bone_index)
            for axis in range(3):
                origin[axis] += translations[bone_index][axis]
            parent = parents[bone_index]
            if parent >= count:
                break
            bone_index = parent
        return tuple(origin)

    return [
        ModelBone(
            bone_index,
            parents[bone_index],
            translations[bone_index],
            world(bone_index),
            channels[bone_index],
        )
        for bone_index in range(count)
    ]


class ModelDecoder:
    def __init__(
        self,
        raw_data: bytes,
        resolve_address: Callable[[int], int | None],
        bones: list[ModelBone] | None = None,
        texture_segments: dict[int, int] | None = None,
        fallback_textures: set[int] | None = None,
    ):
        self.raw_data = raw_data
        self.resolve_address = resolve_address
        self.bones = bones or list()
        self.texture_segments = texture_segments or dict()
        self.fallback_textures = fallback_textures or set()

        self._cache: list[_CacheEntry | None] = [None] * VERTEX_CACHE_SIZE
        self._visited: set[int] = set()
        self._meshes: dict[tuple, _MaterialMesh] = dict()

        self._bone = 0 if self.bones else NO_PARENT
        self._state = _TextureState(prefer_loaded_dimensions=True)
        self._display_list_offset = 0
        self._optional_segment: int | None = None

    @property
    def mesh_groups(self) -> tuple[_MeshGroup, ...]:
        return tuple(
            _MeshGroup(
                vertices=tuple(mesh.vertices),
                triangles=tuple(mesh.triangles),
                texture=mesh.texture,
                display_list_offset=mesh.display_list_offset,
                translucent=mesh.translucent,
                double_sided=mesh.double_sided,
                vertex_bones=tuple(mesh.vertex_bones),
                vertex_normals=tuple(mesh.vertex_normals),
                render=mesh.render,
                optional_segment=mesh.optional_segment,
            )
            for mesh in self._meshes.values()
        )

    def walk(self, start: int | None, depth: int = 0) -> None:
        if start is None or depth > MAX_BRANCH_DEPTH or start in self._visited:
            return
        if not 0 <= start < len(self.raw_data):
            return
        self._visited.add(start)
        self._display_list_offset = start

        position = start
        while position + COMMAND_SIZE <= len(self.raw_data):
            command = get_command(self.raw_data[position : position + COMMAND_SIZE])
            position += COMMAND_SIZE
            if command is None:
                continue

            if isinstance(command, commands.G_VTX):
                self._load_vertices(command)
            elif isinstance(command, commands.G_TRI1):
                self._add_triangle(Triangle.from_tri1(command))
            elif isinstance(command, commands.G_TRI2):
                for triangle in Triangle.from_tri2(command):
                    self._add_triangle(triangle)
            elif isinstance(command, commands.G_QUAD):
                self._add_triangle(Triangle(command.v1, command.v2, command.v3))
                self._add_triangle(
                    Triangle(command.v1_duplicate, command.v3_duplicate, command.v4)
                )
            elif isinstance(command, commands.G_MTX):
                self._load_matrix(command)
            elif isinstance(command, commands.G_DL):
                self._state.apply(command)
                target = self.resolve_address(
                    int.from_bytes(command.segment + command.address, "big")
                )
                if target is not None:
                    self.walk(target, depth + 1)
                    if not command.store_return_address:
                        return
                elif not command.store_return_address:
                    self._optional_segment = command.segment[0]
                self._display_list_offset = start
            elif isinstance(command, commands.G_SPNOOP):
                if int.from_bytes(command.tag, "big") == self._optional_segment:
                    self._optional_segment = None
            elif isinstance(command, commands.G_ENDDL):
                return
            else:
                self._state.apply(command)

    def _load_matrix(self, command: commands.G_MTX) -> None:
        segment, offset = command.address >> 24, command.address & 0xFFFFFF
        if not self.bones or segment != BONE_MATRIX_SEGMENT:
            return
        if offset % MATRIX_SIZE == 0 and offset // MATRIX_SIZE < len(self.bones):
            self._bone = offset // MATRIX_SIZE

    def _load_vertices(self, command: commands.G_VTX) -> None:
        first = command.buffer_start // 2
        count = command.vertex_count
        source = self.resolve_address(
            int.from_bytes(command.segment + command.address, "big")
        )
        if source is None or source < 0 or first < 0 or first + count > VERTEX_CACHE_SIZE:
            return
        for cache_index in range(count):
            offset = source + cache_index * VERTEX_SIZE
            if offset + VERTEX_SIZE > len(self.raw_data):
                break
            self._cache[first + cache_index] = _CacheEntry(
                offset,
                self._bone,
                *self._state.texture_scale,
            )

    def _add_triangle(self, triangle: Triangle) -> None:
        corners = [
            self._cache[index] if index < VERTEX_CACHE_SIZE else None
            for index in (triangle.v1, triangle.v2, triangle.v3)
        ]
        if any(corner is None for corner in corners):
            return

        texture = self._active_texture()
        geometry_mode = self._state.geometry_mode
        translucent = bool(self._state.other_mode_l & G_RM_FORCE_BL)
        double_sided = not geometry_mode & (G_CULL_FRONT | G_CULL_BACK)
        lit = bool(geometry_mode & G_LIGHTING)
        render = self._state.render_state
        key = (
            texture,
            self._display_list_offset,
            translucent,
            double_sided,
            lit,
            render,
            self._optional_segment,
        )
        mesh = self._meshes.get(key)
        if mesh is None:
            mesh = _MaterialMesh(
                texture,
                self._display_list_offset,
                translucent,
                double_sided,
                render,
                self._optional_segment,
            )
            self._meshes[key] = mesh

        indices = [self._mesh_vertex(mesh, corner, texture) for corner in corners]
        if len(set(indices)) == 3:
            mesh.triangles.append(Triangle(*indices))

    def _mesh_vertex(
        self,
        mesh: _MaterialMesh,
        corner: _CacheEntry,
        texture: _TextureKey | None,
    ) -> int:
        shading = self._state.geometry_mode & (G_LIGHTING | G_TEXTURE_GEN)
        key = (corner, self._state.texture_tile, shading, texture)
        index = mesh.indices.get(key)
        if index is not None:
            return index

        index = len(mesh.vertices)
        mesh.indices[key] = index
        mesh.vertices.append(self._vertex(corner, texture))
        mesh.vertex_bones.append(0 if corner.bone == NO_PARENT else corner.bone)
        if self._state.geometry_mode & G_LIGHTING:
            mesh.vertex_normals.append(self._normal(corner))
        return index

    def _normal(self, corner: _CacheEntry) -> tuple[float, float, float]:
        axes = [
            byte - 0x100 if byte > 0x7F else byte
            for byte in self.raw_data[corner.offset + 12 : corner.offset + 15]
        ]
        length = math.sqrt(sum(axis * axis for axis in axes))
        if length == 0:
            return (0.0, 1.0, 0.0)
        return tuple(axis / length for axis in axes)

    def _vertex(self, corner: _CacheEntry, texture: _TextureKey | None) -> Vertex:
        vertex = Vertex.from_bytes(
            self.raw_data[corner.offset : corner.offset + VERTEX_SIZE]
        )
        origin = (0.0, 0.0, 0.0)
        if corner.bone != NO_PARENT and corner.bone < len(self.bones):
            origin = self.bones[corner.bone].world
        geometry_mode = self._state.geometry_mode
        if geometry_mode & G_TEXTURE_GEN and texture is not None:
            normal = self._normal(corner)
            texture_cord_u = round((0.5 + normal[0] / 2) * texture.width * 32)
            texture_cord_v = round((0.5 - normal[1] / 2) * texture.height * 32)
            white = True
        else:
            shift_s, shift_t = self._state.texture_tile_shift
            uls, ult = self._state.texture_tile_origin
            texture_cord_u = (
                round(_signed_16(vertex.texture_cord_u) * corner.scale_s * shift_scale(shift_s))
                - uls * 8
            )
            texture_cord_v = (
                round(_signed_16(vertex.texture_cord_v) * corner.scale_t * shift_scale(shift_t))
                - ult * 8
            )
            white = bool(geometry_mode & G_LIGHTING)
        return replace(
            vertex,
            x=round(vertex.x + origin[0]),
            y=round(vertex.y + origin[1]),
            z=round(vertex.z + origin[2]),
            texture_cord_u=_clamp_texel(texture_cord_u),
            texture_cord_v=_clamp_texel(texture_cord_v),
            xr=0xFF if white else vertex.xr,
            yg=0xFF if white else vertex.yg,
            zb=0xFF if white else vertex.zb,
        )

    def _active_texture(self) -> _TextureKey | None:
        if not self._state.combiner_reads_texture:
            return None
        texture = self._state.active_texture
        if texture is None:
            return None

        image_index = texture.image_index
        if image_index >> 24:
            image_index = self.texture_segments.get(image_index >> 24)
            if image_index is None:
                return None
        elif image_index not in self.fallback_textures:
            return texture

        return replace(
            texture,
            image_index=image_index,
            from_fallback_table=image_index in self.fallback_textures,
        )


def _clamp_texel(texel: int) -> int:
    return max(-0x8000, min(0x7FFF, texel)) & 0xFFFF
